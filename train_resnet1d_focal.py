import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

from shared_utils import setup_experiment, load_data, ECGDataset, FocalLoss

# --- MODEL DEFINITION ---

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet1DECG(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet1DECG, self).__init__()
        self.in_channels = 32
        self.conv = nn.Conv1d(1, self.in_channels, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.meta_fc = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 32), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(256 + 32, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, sig, meta):
        x = self.pool(self.relu(self.bn(self.conv(sig))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).view(x.size(0), -1)
        m = self.meta_fc(meta)
        return self.fc(torch.cat((x, m), dim=1))


def train():
    exp_dir, logger = setup_experiment('ResNet1D')
    
    # You MUST place the dataset in a folder named 'dataset' next to the script,
    # or update this path on your machines.
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
    
    BATCH_SIZE = 256
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")
    
    class_counts = np.bincount(y)
    weights = 1.0 / class_counts
    weights = weights / weights.sum()
    alpha = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    model = ResNet1DECG().to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    best_f1 = 0.0
    
    logger.info("Starting training...")
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for sigs, metas, labels in train_loader:
            sigs, metas, labels = sigs.to(DEVICE), metas.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(sigs, metas)
            loss = criterion(outputs, labels)
            loss.backward()
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
                outputs = model(sigs, metas)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * sigs.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        logger.info(f"Epoch {epoch+1}/30 - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))
            logger.info(f" -> Saved new best model (F1: {best_f1:.4f})")

if __name__ == '__main__':
    train()
