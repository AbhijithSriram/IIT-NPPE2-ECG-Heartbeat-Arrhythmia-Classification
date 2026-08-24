import json

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + '\n' for line in text.split('\n')]})

def add_code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + '\n' for line in text.split('\n')]})

add_md("# \U0001F9EA Domain-Adversarial Ensemble (DANN CNN + Covariate-Shift-Weighted XGBoost)")
add_md("""**Escalation from the previous submission (public LB 0.546).** That notebook fixed part
of the gap with RR-feature emphasis + augmentation + block-CV, up from 0.513/0.516, but a
large gap to local CV clearly remained.

We went and directly measured whether train.csv and test.csv come from different
distributions: trained a plain classifier to predict "is this beat from train or test"
using only engineered features. It scored **0.996 AUC** -- train and test are almost
perfectly separable. The top two drivers were `pre_rr` and `post_pre_ratio` (raw RR
timing), not signal shape. That means part of what the previous notebook added as
"patient-invariant" RR features (built from raw `pre_rr`/`post_rr`, not properly
recording-normalized) was itself carrying shift.

**This notebook does three new things:**

1. **Fixes the RR normalization.** The host already gives us `rr_ratio = pre_rr / median_RR_of_that_recording`.
   We invert that to recover an estimate of each beat's recording-level median RR
   (`implied_median_rr = pre_rr / rr_ratio`), then express `post_rr` and every derived
   RR feature relative to *that*, instead of raw seconds -- removing each recording's
   baseline heart-rate scale, which is exactly what was still leaking.
2. **Domain-Adversarial Training (DANN)** for the CNN: a second head predicts train-vs-test
   from the same learned features, through a Gradient Reversal Layer, so the feature
   extractor is explicitly penalized for encoding whatever makes train and test
   distinguishable -- forcing it toward representations that generalize across the
   shift instead of memorizing per-recording quirks.
3. **Covariate-shift-weighted XGBoost.** We reuse the train-vs-test classifier's
   out-of-fold probability as an importance weight (tempered to avoid blow-up under
   near-perfect separability), so training examples that "look more like test" get
   upweighted.

Both use unlabeled `test.csv` inputs only for domain adaptation -- no test labels are
touched anywhere, and no pretrained weights or external data are used.""")

add_md("## 0. Imports")
add_code("""import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")""")

add_md("## 1. Setup & Data Loading")
add_code("""KAGGLE_INPUT_DIR = '/kaggle/input/nppe-2-t-2-26-ecg-heartbeat-arrhythmia-classification/nppe2_dataset'
TRAIN_PATH = os.path.join(KAGGLE_INPUT_DIR, 'train.csv')
TEST_PATH = os.path.join(KAGGLE_INPUT_DIR, 'test.csv')

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = 'dataset/train.csv'
    TEST_PATH = 'dataset/test.csv'

print("Loading training and testing data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

SIG_COLS = [f'sig_{i}' for i in range(250)]""")

add_md("""## 2. Recording-Normalized RR Features

`rr_ratio` is already `pre_rr / median_RR_of_the_recording` (host-provided). We invert it
to recover each beat's implied recording-level median RR, then re-express every other RR
quantity relative to that -- instead of raw seconds, which differ by each patient's resting
heart rate and turned out to be a top driver of the train/test shift.""")
add_code("""def add_rr_features(df):
    df = df.copy()
    for col in ['pre_rr', 'post_rr', 'rr_ratio']:
        df[col] = df[col].fillna(df[col].median())
    # avoid divide-by-zero on rr_ratio
    safe_ratio = df['rr_ratio'].replace(0, np.nan).fillna(df['rr_ratio'].median())
    implied_median_rr = df['pre_rr'] / safe_ratio
    implied_median_rr = implied_median_rr.replace([np.inf, -np.inf], np.nan)
    implied_median_rr = implied_median_rr.fillna(implied_median_rr.median())
    implied_median_rr = implied_median_rr.clip(lower=1e-3)

    df['implied_median_rr'] = implied_median_rr
    df['post_rr_norm'] = df['post_rr'] / implied_median_rr          # recording-scale-free
    df['rr_diff_norm'] = df['post_rr_norm'] - df['rr_ratio']         # both already recording-relative
    df['rr_sum_norm'] = df['post_rr_norm'] + df['rr_ratio']
    df['prematurity'] = 1.0 - df['rr_ratio']                         # positive => early beat
    df['post_pre_ratio_norm'] = df['post_rr_norm'] / (df['rr_ratio'] + 1e-6)  # compensatory-pause signature
    return df

train_df = add_rr_features(train_df)
test_df = add_rr_features(test_df)

RR_META_COLS = ['rr_ratio', 'post_rr_norm', 'rr_diff_norm', 'rr_sum_norm',
                'prematurity', 'post_pre_ratio_norm']
print("Using recording-normalized RR features:", RR_META_COLS)
print("(raw pre_rr / post_rr / seconds-scale features deliberately excluded -- they were")
print(" the #1 and #2 drivers of train/test distinguishability, see intro.)")""")

add_md("## 3. Statistical / Frequency Signal Features (for XGBoost)")
add_code("""def extract_stat_features(df):
    signals = df[SIG_COLS].values.astype(np.float64)
    d1 = np.diff(signals, axis=1)
    d2 = np.diff(d1, axis=1)
    feats = np.column_stack([
        signals.mean(axis=1), signals.std(axis=1), signals.max(axis=1), signals.min(axis=1),
        np.median(signals, axis=1), skew(signals, axis=1), kurtosis(signals, axis=1),
        np.sum(signals ** 2, axis=1),
        np.sum(np.diff(np.sign(signals), axis=1) != 0, axis=1),
        d1.mean(axis=1), d1.std(axis=1), d1.max(axis=1), d1.min(axis=1),
        d2.mean(axis=1), d2.std(axis=1), d2.max(axis=1), d2.min(axis=1),
        np.abs(np.fft.rfft(signals, axis=1))[:, 1:11],
    ])
    feat_df = pd.DataFrame(feats)  # positional names, consistent with the RR cols appended below
    for col in RR_META_COLS:
        feat_df[col] = df[col].values
    return feat_df


def normalize_signals(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1e-8
    return (X - means) / stds


print("Extracting statistical features...")
X_stat_train = extract_stat_features(train_df)
X_stat_test = extract_stat_features(test_df)
y_train = train_df['label'].values
test_ids = test_df['id'].values
print(f"Extracted {X_stat_train.shape[1]} features.")

print("Normalizing raw signals (for the CNN)...")
X_sig_train = normalize_signals(train_df[SIG_COLS].values.astype(np.float32))
X_sig_test = normalize_signals(test_df[SIG_COLS].values.astype(np.float32))

meta_scaler = StandardScaler()
X_meta_train = meta_scaler.fit_transform(train_df[RR_META_COLS].values.astype(np.float32))
X_meta_test = meta_scaler.transform(test_df[RR_META_COLS].values.astype(np.float32))""")

add_md("""## 4. Confirm the Shift, and Derive Importance Weights

Re-measure train-vs-test distinguishability with the fixed (normalized) feature set, then
use the resulting out-of-fold domain probability as a *tempered* importance weight for
XGBoost -- upweighting train examples whose features look more "test-like". Weights are
clipped and power-tempered because under near-perfect separability, raw `p/(1-p)`
importance weights explode.""")
add_code("""X_domain_all = pd.concat([X_stat_train, X_stat_test], axis=0, ignore_index=True)
y_domain = np.concatenate([np.zeros(len(X_stat_train)), np.ones(len(X_stat_test))])

skf_dom = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_domain_prob = np.zeros(len(X_domain_all))
dom_params = dict(objective='binary:logistic', eval_metric='auc', max_depth=6, learning_rate=0.08,
                   subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                   device='cuda' if DEVICE.type == 'cuda' else 'cpu', random_state=SEED, n_estimators=300)

for tr_idx, va_idx in skf_dom.split(X_domain_all, y_domain):
    clf = xgb.XGBClassifier(**dom_params)
    clf.fit(X_domain_all.iloc[tr_idx], y_domain[tr_idx], verbose=False)
    oof_domain_prob[va_idx] = clf.predict_proba(X_domain_all.iloc[va_idx])[:, 1]

domain_auc = roc_auc_score(y_domain, oof_domain_prob)
print(f"Train-vs-test domain AUC with normalized RR features: {domain_auc:.4f}")
print("(compare to 0.9964 with the raw/unnormalized RR features used previously)")

# importance weights for the TRAIN rows only: p(test) / p(train), tempered + clipped
p_test_train_rows = np.clip(oof_domain_prob[:len(X_stat_train)], 0.02, 0.98)
raw_weight = p_test_train_rows / (1.0 - p_test_train_rows)
TEMPERATURE = 0.5  # <1 softens extreme weights
domain_weight = raw_weight ** TEMPERATURE
domain_weight = np.clip(domain_weight, 0.2, 5.0)
domain_weight = domain_weight / domain_weight.mean()  # keep overall scale ~1
print(f"Domain importance weight stats: min={domain_weight.min():.3f} "
      f"median={np.median(domain_weight):.3f} max={domain_weight.max():.3f}")""")

add_md("""## 5. Block/Group Validation Split (unchanged approach)

Same pseudo-recording clustering as the previous notebook, used only for local
validation -- explicitly not leaderboard-comparable, just a directional, less-leaky
signal than a plain random split.""")
add_code("""print("Building pseudo-groups for block CV...")
pca = PCA(n_components=30, random_state=SEED)
sig_pca = pca.fit_transform(X_sig_train)
rr_for_cluster = StandardScaler().fit_transform(train_df[['rr_ratio', 'post_rr_norm']].values)
cluster_features = np.concatenate([sig_pca, rr_for_cluster], axis=1)

N_CLUSTERS = 400
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10, batch_size=1024)
pseudo_groups = kmeans.fit_predict(cluster_features)
print(f"Built {len(np.unique(pseudo_groups))} pseudo-groups, "
      f"mean size {len(train_df) / len(np.unique(pseudo_groups)):.1f}")

N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)
FOLD_SPLITS = list(gkf.split(X_stat_train, y_train, groups=pseudo_groups))
for i, (tr, va) in enumerate(FOLD_SPLITS):
    print(f"  fold {i}: train={len(tr)} val={len(va)}")""")

add_md("""## 6. DANN Model: ResNet1D backbone + Gradient-Reversal domain head

Same residual-CNN backbone family as before. New: a domain classifier head reads the
same pooled features through a `GradientReversalLayer` -- forward pass is identity,
backward pass negates (and scales) the gradient, so gradient descent on the domain loss
becomes gradient *ascent* for the feature extractor: it is pushed toward features the
domain head *cannot* use to tell train from test apart, while the class head still needs
those same features to separate arrhythmia types. The adversarial strength `lambda_` is
ramped up over training (standard DANN schedule) so the network first learns to classify
before the adversarial pressure kicks in hard.""")
add_code("""class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 0.0

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class DANNResNet1D(nn.Module):
    def __init__(self, num_classes=4, n_meta=6, meta_hidden=64, dropout=0.45):
        super().__init__()
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

        self.meta_fc = nn.Sequential(
            nn.Linear(n_meta, meta_hidden), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(meta_hidden, meta_hidden), nn.ReLU(),
        )
        feat_dim = 256 + meta_hidden

        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels))
        layers = [ResidualBlock1D(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def features(self, sig, meta):
        x = self.pool(self.relu(self.bn(self.conv(sig))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).view(x.size(0), -1)
        m = self.meta_fc(meta)
        return torch.cat((x, m), dim=1)

    def forward(self, sig, meta):
        feat = self.features(sig, meta)
        class_logits = self.class_head(feat)
        domain_logits = self.domain_head(self.grl(feat)).squeeze(1)
        return class_logits, domain_logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


class ECGDataset(Dataset):
    def __init__(self, signals, metas, labels=None):
        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
        self.metas = torch.tensor(metas, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.signals[idx], self.metas[idx], self.labels[idx]
        return self.signals[idx], self.metas[idx]


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


# --- augmentation (train-time only), same family as the previous notebook ---
def aug_noise(x, level=0.02):
    return x + torch.randn_like(x) * level

def aug_scale_shift(x, scale_range=(0.85, 1.15), shift_range=0.1):
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(*scale_range)
    shift = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(-shift_range, shift_range)
    return x * scale + shift

def aug_time_shift(x, max_shift=10):
    shift = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
    return torch.roll(x, shifts=shift, dims=2)

def aug_baseline_wander(x, device, max_amp=0.15):
    b, _, L = x.shape
    t = torch.linspace(0, 1, L, device=device).view(1, 1, L)
    freq = torch.empty(b, 1, 1, device=device).uniform_(0.5, 2.0)
    phase = torch.empty(b, 1, 1, device=device).uniform_(0, 2 * np.pi)
    amp = torch.empty(b, 1, 1, device=device).uniform_(0, max_amp)
    return x + amp * torch.sin(2 * np.pi * freq * t + phase)

def train_augment(sigs, device):
    sigs = aug_time_shift(sigs)
    sigs = aug_baseline_wander(sigs, device)
    sigs = aug_scale_shift(sigs)
    sigs = aug_noise(sigs)
    return sigs""")

add_md("## 7. Train the DANN CNN with GroupKFold (5 folds)")
add_code("""CNN_EPOCHS = 30
CNN_PATIENCE = 8
BATCH_SIZE = 256
MAX_LAMBDA = 0.6  # peak adversarial strength

class_counts = np.bincount(y_train)
alpha = torch.tensor(1.0 / class_counts, dtype=torch.float32)
alpha = (alpha / alpha.sum()).to(DEVICE)

# unlabeled domain pool: ALL of test.csv, reused across every fold (fixed adaptation target)
test_domain_ds = ECGDataset(X_sig_test, X_meta_test)
test_domain_loader = DataLoader(test_domain_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

oof_cnn_probs = np.zeros((len(train_df), 4), dtype=np.float32)
cnn_models_state = []
cnn_fold_f1 = []
_cnn_start = time.time()

for fold, (tr_idx, va_idx) in enumerate(FOLD_SPLITS):
    print(f"\\n=== DANN CNN fold {fold} === ({time.time() - _cnn_start:.0f}s elapsed so far)")
    train_ds = ECGDataset(X_sig_train[tr_idx], X_meta_train[tr_idx], y_train[tr_idx])
    val_ds = ECGDataset(X_sig_train[va_idx], X_meta_train[va_idx], y_train[va_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    domain_iter = infinite_loader(test_domain_loader)

    model = DANNResNet1D(num_classes=4, n_meta=len(RR_META_COLS)).to(DEVICE)
    class_criterion = FocalLoss(alpha=alpha, gamma=2.0)
    domain_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS, eta_min=1e-6)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CNN_EPOCHS
    global_step = 0

    best_f1, best_state, no_improve = 0.0, None, 0
    for epoch in range(CNN_EPOCHS):
        model.train()
        domain_acc_meter = []
        for sigs, metas, labels in train_loader:
            p = global_step / max(total_steps, 1)
            lambda_ = MAX_LAMBDA * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
            model.grl.lambda_ = lambda_

            sigs, metas, labels = sigs.to(DEVICE), metas.to(DEVICE), labels.to(DEVICE)
            sigs = train_augment(sigs, DEVICE)

            dom_sigs, dom_metas = next(domain_iter)
            dom_sigs, dom_metas = dom_sigs.to(DEVICE), dom_metas.to(DEVICE)
            dom_sigs = train_augment(dom_sigs, DEVICE)

            combined_sig = torch.cat([sigs, dom_sigs], dim=0)
            combined_meta = torch.cat([metas, dom_metas], dim=0)
            domain_labels = torch.cat([
                torch.zeros(sigs.size(0), device=DEVICE),
                torch.ones(dom_sigs.size(0), device=DEVICE),
            ])

            optimizer.zero_grad()
            class_logits, domain_logits = model(combined_sig, combined_meta)
            class_loss = class_criterion(class_logits[:sigs.size(0)], labels)
            domain_loss = domain_criterion(domain_logits, domain_labels)
            loss = class_loss + domain_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                dom_preds = (torch.sigmoid(domain_logits) > 0.5).float()
                domain_acc_meter.append((dom_preds == domain_labels).float().mean().item())
            global_step += 1
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sigs, metas, labels in val_loader:
                sigs, metas = sigs.to(DEVICE), metas.to(DEVICE)
                class_logits, _ = model(sigs, metas)
                preds = torch.argmax(class_logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        val_f1 = f1_score(all_labels, all_preds, average='macro')

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= CNN_PATIENCE:
                print(f"  early stop at epoch {epoch+1} (best F1={best_f1:.4f})")
                break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{CNN_EPOCHS} - val macro F1: {val_f1:.4f} (best: {best_f1:.4f}) "
                  f"- lambda: {lambda_:.3f} - domain acc: {np.mean(domain_acc_meter):.3f} "
                  f"(0.50 = fully confused, good; 1.00 = fully distinguishable, bad)")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = []
        for sigs, metas, _ in val_loader:
            sigs, metas = sigs.to(DEVICE), metas.to(DEVICE)
            class_logits, _ = model(sigs, metas)
            probs.append(torch.softmax(class_logits, dim=1).cpu().numpy())
        oof_cnn_probs[va_idx] = np.concatenate(probs)

    cnn_fold_f1.append(best_f1)
    cnn_models_state.append(best_state)
    print(f"DANN CNN fold {fold} best macro F1: {best_f1:.4f}")

cnn_oof_f1 = f1_score(y_train, np.argmax(oof_cnn_probs, axis=1), average='macro')
print(f"\\nDANN CNN training took {time.time() - _cnn_start:.0f}s total")
print(f"DANN CNN block-CV OOF macro F1: {cnn_oof_f1:.4f} (fold mean {np.mean(cnn_fold_f1):.4f})")
print(classification_report(y_train, np.argmax(oof_cnn_probs, axis=1), digits=4))""")

add_md("## 8. Train covariate-shift-weighted XGBoost on the same folds")
add_code("""xgb_params = dict(
    objective='multi:softprob', num_class=4, eval_metric='mlogloss',
    max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=2.0, reg_alpha=0.5,
    tree_method='hist', device='cuda' if DEVICE.type == 'cuda' else 'cpu',
    random_state=SEED, n_estimators=600, early_stopping_rounds=40,
)

oof_xgb_probs = np.zeros((len(train_df), 4))
xgb_models = []
xgb_fold_f1 = []
class_balance_weights = compute_sample_weight('balanced', y_train)
# combine class-balance weighting with the covariate-shift (domain) weighting from Section 4
combined_weights = class_balance_weights * domain_weight

for fold, (tr_idx, va_idx) in enumerate(FOLD_SPLITS):
    X_t, y_t = X_stat_train.iloc[tr_idx], y_train[tr_idx]
    X_v, y_v = X_stat_train.iloc[va_idx], y_train[va_idx]
    w_t = combined_weights[tr_idx]

    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X_t, y_t, sample_weight=w_t, eval_set=[(X_v, y_v)], verbose=False)

    probs = clf.predict_proba(X_v)
    oof_xgb_probs[va_idx] = probs
    fold_f1 = f1_score(y_v, np.argmax(probs, axis=1), average='macro')
    xgb_fold_f1.append(fold_f1)
    xgb_models.append(clf)
    print(f"XGB fold {fold}: macro F1 = {fold_f1:.4f}")

xgb_oof_f1 = f1_score(y_train, np.argmax(oof_xgb_probs, axis=1), average='macro')
print(f"\\nXGB block-CV OOF macro F1: {xgb_oof_f1:.4f} (fold mean {np.mean(xgb_fold_f1):.4f})")
print(classification_report(y_train, np.argmax(oof_xgb_probs, axis=1), digits=4))""")

add_md("## 9. OOF Ensemble Check")
add_code("""for w_cnn in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = w_cnn * oof_cnn_probs + (1 - w_cnn) * oof_xgb_probs
    blend_f1 = f1_score(y_train, np.argmax(blend, axis=1), average='macro')
    print(f"w_cnn={w_cnn:.1f}: block-CV OOF ensemble macro F1 = {blend_f1:.4f}")

W_CNN = 0.5
print(f"\\nUsing w_cnn={W_CNN} for the final test ensemble (adjust based on the sweep above).")""")

add_md("## 10. Inference & Submission")
add_code("""def predict_cnn_test(states, X_sig, X_meta, device):
    ds = ECGDataset(X_sig, X_meta)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    all_probs = np.zeros((len(X_sig), 4), dtype=np.float32)
    model = DANNResNet1D(num_classes=4, n_meta=X_meta.shape[1]).to(device)
    for state in states:
        model.load_state_dict(state)
        model.eval()
        probs = []
        with torch.no_grad():
            for sigs, metas in loader:
                sigs, metas = sigs.to(device), metas.to(device)
                out_clean, _ = model(sigs, metas)
                out_noisy, _ = model(aug_noise(sigs, 0.02), metas)
                probs.append(((torch.softmax(out_clean, dim=1) + torch.softmax(out_noisy, dim=1)) / 2).cpu().numpy())
        all_probs += np.concatenate(probs) / len(states)
    return all_probs

print("Running DANN CNN inference on test.csv...")
cnn_test_probs = predict_cnn_test(cnn_models_state, X_sig_test, X_meta_test, DEVICE)

print("Running XGBoost inference on test.csv...")
xgb_test_probs = np.zeros((len(test_df), 4))
for clf in xgb_models:
    xgb_test_probs += clf.predict_proba(X_stat_test) / len(xgb_models)

ensemble_test_probs = W_CNN * cnn_test_probs + (1 - W_CNN) * xgb_test_probs
final_preds = np.argmax(ensemble_test_probs, axis=1)

print("\\nPredicted label distribution on test.csv:")
print(pd.Series(final_preds).value_counts(normalize=True).sort_index())
print("\\nTrain label distribution (for comparison):")
print(pd.Series(y_train).value_counts(normalize=True).sort_index())
print("\\nPrior submissions predicted ~85-88% Normal / ~9-14% class-2 (Ventricular Ectopic)")
print("vs train's 62.7% / 31.4% -- the gap above should be visibly smaller if this run helped.")

submission_df = pd.DataFrame({'id': test_ids, 'label': final_preds})
submission_df.to_csv('submission.csv', index=False)
print("\\nSubmission saved to submission.csv")
submission_df.head()""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("submission_dann.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Wrote submission_dann.ipynb")
