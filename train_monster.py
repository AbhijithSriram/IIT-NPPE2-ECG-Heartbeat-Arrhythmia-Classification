import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import math

from shared_utils import setup_experiment, load_data, ECGDataset, FocalLoss

# --- Stochastic Depth ---
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# --- Squeeze-and-Excitation ---
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# --- ConvNeXt-1D Block ---
class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=11, padding=5, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.se = SEBlock1D(dim, reduction=4)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)
        x = self.se(x)
        x = input + self.drop_path(x)
        return x

class MonsterECG(nn.Module):
    def __init__(self, num_classes=4, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.4):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv1d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(dims[i+1])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock1D(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        self.meta_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.GELU()
        )
        
        self.head = nn.Sequential(
            nn.Linear(dims[-1] + 64, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, sig, meta):
        x = sig
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            
        x = x.mean(dim=2)
        x = self.norm(x)
        m = self.meta_fc(meta)
        return self.head(torch.cat((x, m), dim=1))

# --- Mixup ---
def mixup_data(x, meta, y, alpha=0.2, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_meta = lam * meta + (1 - lam) * meta[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_meta, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- Augmentations ---
def add_noise(x, noise_level=0.01):
    return x + torch.randn_like(x) * noise_level
    
def random_scale(x, scale_range=(0.8, 1.2)):
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(*scale_range)
    return x * scale

# --- Scheduler ---
from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1, min_lr=0.001, warmup_steps=0, gamma=1.0, last_epoch=-1):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        self.init_lr()
        
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
            
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult**(n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# --- Main Training ---
def train():
    exp_dir, logger = setup_experiment('Monster_SE_ConvNeXt1D')
    train_path = 'dataset/train.csv'
    
    if not os.path.exists(train_path):
        logger.error(f"Dataset not found at {train_path}. Please create a 'dataset' folder and place train.csv in it.")
        return
        
    X_sig, X_time, y = load_data(train_path)
    
    X_sig_t, X_sig_v, X_time_t, X_time_v, y_t, y_v = train_test_split(
        X_sig, X_time, y, test_size=0.2, stratify=y, random_state=42
    )
    
    train_dataset = ECGDataset(X_sig_t, X_time_t, y_t)
    val_dataset = ECGDataset(X_sig_v, X_time_v, y_v)
    
    # Weighted Random Sampler
    class_counts = np.bincount(y_t)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_t]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")
    
    model = MonsterECG(num_classes=4).to(DEVICE)
    criterion = FocalLoss(gamma=2.0)
    
    EPOCHS = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=EPOCHS, max_lr=1e-3, min_lr=1e-6, warmup_steps=10)
    
    best_f1 = 0.0
    
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for sigs, metas, labels in train_loader:
            sigs, metas, labels = sigs.to(DEVICE), metas.to(DEVICE), labels.to(DEVICE)
            
            sigs = add_noise(sigs)
            sigs = random_scale(sigs)
            
            sigs, metas, targets_a, targets_b, lam = mixup_data(sigs, metas, labels, alpha=0.4, device=DEVICE)
            
            optimizer.zero_grad()
            outputs = model(sigs, metas)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * sigs.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for sigs, metas, labels in val_loader:
                sigs, metas, labels = sigs.to(DEVICE), metas.to(DEVICE), labels.to(DEVICE)
                
                # Test-Time Augmentation (TTA)
                out_orig = model(sigs, metas)
                out_n1 = model(add_noise(sigs, 0.01), metas)
                out_s1 = model(random_scale(sigs, (0.9, 0.9)), metas)
                out_s2 = model(random_scale(sigs, (1.1, 1.1)), metas)
                
                outputs = (out_orig + out_n1 + out_s1 + out_s2) / 4.0
                
                loss = criterion(outputs, labels)
                val_loss += loss.item() * sigs.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        logger.info(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            logger.info(f" -> Saved new best model (F1: {best_f1:.4f})")

if __name__ == '__main__':
    train()
