import json

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + '\n' for line in text.split('\n')]})

def add_code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + '\n' for line in text.split('\n')]})

add_md("# \U0001F3E5 Patient-Robust Ensemble (CNN + XGBoost, block-validated)")
add_md("""**Why this notebook exists.** Two prior submissions (a raw-signal ConvNeXt1D CNN and a
hand-engineered-feature XGBoost model) both scored ~0.92-0.94 on local CV
(`train_test_split`/`StratifiedKFold(shuffle=True)`) but only ~0.51 on the public
leaderboard, for both models, despite very different architectures.

We diagnosed this (see `diagnostics/FINDINGS.md` in the repo): running the actual trained
models on `test.csv` and comparing predicted class proportions to train's true label
distribution showed **both models collapse toward predicting "Normal" and drastically
under-predict Ventricular Ectopic beats** (9-14% predicted vs. 31.4% in train). That's the
classic inter-patient generalization failure documented in ECG arrhythmia literature: a
model validated on a random row-level split effectively memorizes per-patient beat
morphology, which does not transfer to a genuinely new patient's beat shape. We also
confirmed a plain literal-near-duplicate-beat leakage theory and a prior/calibration-shift
theory both fail to explain the gap on their own.

**What this notebook does differently:**
1. Leans harder on `rr_ratio` and derived RR-interval features, which the dataset host
   already normalizes per-recording (`rr_ratio` = pre_rr / median RR of *that* recording)
   -- these are physiologically patient-invariant, unlike raw QRS/beat shape.
2. Adds ECG-realistic augmentation (baseline wander, time-shift, amplitude jitter, noise,
   mixup) that discourages the CNN from memorizing an exact per-patient waveform template.
3. Uses **block/group cross-validation** (beats clustered into pseudo-groups instead of a
   plain random split) as a more conservative, less-leaky local validation signal. We are
   explicit that this is an imperfect proxy for a true per-patient split (we don't have
   patient IDs) -- treat the resulting CV number as *directional*, not as leaderboard-
   comparable.
4. Ensembles a regularized CNN (raw signal) with an XGBoost model (engineered features),
   since the two prior submissions' predictions only agreed on ~90% of test beats -- there
   is a real disagreement zone an ensemble can arbitrate.

No pretrained weights or transfer learning are used anywhere in this notebook.""")

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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

import xgboost as xgb

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")""")

add_md("## 1. Setup & Data Loading")
add_code("""# Paths in Kaggle Environment
KAGGLE_INPUT_DIR = '/kaggle/input/nppe-2-t-2-26-ecg-heartbeat-arrhythmia-classification/nppe2_dataset'
TRAIN_PATH = os.path.join(KAGGLE_INPUT_DIR, 'train.csv')
TEST_PATH = os.path.join(KAGGLE_INPUT_DIR, 'test.csv')

# Fallback for local execution
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = 'dataset/train.csv'
    TEST_PATH = 'dataset/test.csv'

print("Loading training and testing data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")

SIG_COLS = [f'sig_{i}' for i in range(250)]""")

add_md("## 2. EDA (quick)")
add_code("""print("Class distribution (train):")
print(train_df['label'].value_counts(normalize=True).sort_index())
print(train_df['label'].value_counts().sort_index())

print("\\nMissing values in timing columns:")
print(train_df[['pre_rr', 'post_rr', 'rr_ratio']].isnull().sum())

print("\\nrr_ratio by class (this is host-normalized: pre_rr / median RR of that recording,")
print("so it is largely patient-invariant -- notice how it separates ectopic (1,2) from Normal (0)):")
print(train_df.groupby('label')['rr_ratio'].describe()[['mean', 'std', 'min', 'max']])""")

add_md("""## 3. Feature Engineering

`rr_ratio` already removes each recording's baseline heart-rate scale, but we add a few
more *ratio/difference*-based RR features -- these stay meaningful regardless of a
patient's resting heart rate, unlike raw beat morphology:

- `prematurity` = `1 - rr_ratio` (positive means the beat arrived early -- the classic
  ectopic-beat signature)
- `post_pre_ratio` = `post_rr / pre_rr` (captures the "compensatory pause" that often
  follows a ventricular ectopic beat: a short pre_rr followed by an unusually long
  post_rr)

We also keep the same statistical/derivative/frequency signal features used by the prior
XGBoost submission (mean, std, skew, kurtosis, energy, zero-crossings, 1st/2nd derivative
stats, top FFT magnitudes) -- vectorized here for speed.""")
add_code("""def add_rr_features(df):
    df = df.copy()
    for col in ['pre_rr', 'post_rr', 'rr_ratio']:
        df[col] = df[col].fillna(df[col].median())
    df['rr_diff'] = df['post_rr'] - df['pre_rr']
    df['rr_sum'] = df['post_rr'] + df['pre_rr']
    df['rr_norm_pre'] = df['pre_rr'] / (df['rr_sum'] + 1e-6)
    df['prematurity'] = 1.0 - df['rr_ratio']
    df['post_pre_ratio'] = df['post_rr'] / (df['pre_rr'] + 1e-6)
    return df

RR_META_COLS = ['pre_rr', 'post_rr', 'rr_ratio', 'rr_diff', 'rr_sum',
                'rr_norm_pre', 'prematurity', 'post_pre_ratio']

train_df = add_rr_features(train_df)
test_df = add_rr_features(test_df)


def extract_stat_features(df):
    \"\"\"Vectorized statistical/derivative/frequency features (same definitions as the
    prior XGBoost submission's extract_features, just without the slow per-row loop).\"\"\"
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
    feat_df = pd.DataFrame(feats)  # positional column names, same convention as before
    for col in RR_META_COLS:
        feat_df[col] = df[col].values
    return feat_df


def normalize_signals(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1e-8
    return (X - means) / stds


print("Extracting statistical features (for XGBoost)...")
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

add_md("""## 4. Block/Group Validation Split

We don't have true patient/recording IDs, so we approximate them: cluster beats into a
moderate number of groups (by raw-signal PCA + RR-interval similarity) and validate with
`GroupKFold` on those pseudo-groups instead of a plain random split. This is deliberately
*coarser* than exact near-duplicate matching -- it forces each fold's validation beats to
come from a different neighborhood of the feature space than its training beats, which is
a much closer (if imperfect) analogue of "a genuinely new patient" than a random split
gives us.

**Caveat we want to be explicit about:** this is a best-effort proxy, not ground truth.
Treat the resulting macro-F1 numbers below as a sanity check and a way to compare this
notebook's own configurations against each other -- not as a leaderboard estimate.""")
add_code("""print("Building pseudo-groups for block CV...")
pca = PCA(n_components=30, random_state=SEED)
sig_pca = pca.fit_transform(X_sig_train)
rr_for_cluster = StandardScaler().fit_transform(train_df[['pre_rr', 'post_rr', 'rr_ratio']].values)
cluster_features = np.concatenate([sig_pca, rr_for_cluster], axis=1)

N_CLUSTERS = 400
kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10, batch_size=1024)
pseudo_groups = kmeans.fit_predict(cluster_features)
print(f"Built {len(np.unique(pseudo_groups))} pseudo-groups, "
      f"mean size {len(train_df) / len(np.unique(pseudo_groups)):.1f}")

N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)
# materialize once so the CNN and XGBoost use the EXACT same folds -- required for a
# valid apples-to-apples OOF ensemble comparison later.
FOLD_SPLITS = list(gkf.split(X_stat_train, y_train, groups=pseudo_groups))
for i, (tr, va) in enumerate(FOLD_SPLITS):
    print(f"  fold {i}: train={len(tr)} val={len(va)}")""")

add_md("""## 5. Model A: Regularized 1D ResNet + RR meta branch

Same residual-CNN backbone family as the earlier (unsuccessful) submissions, but trained
differently:
- Heavier dropout and weight decay.
- On-the-fly augmentation designed to break exact morphology memorization: Gaussian
  noise, amplitude scale/shift jitter, small circular time-shift (R-peak alignment is
  never perfectly centered across recordings), synthetic baseline wander, and mixup.
- The 8 RR-derived meta features (Section 3) get a wider branch than in prior notebooks,
  so the classifier head has real capacity to lean on them.""")
add_code("""class ResidualBlock1D(nn.Module):
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


class RobustResNet1D(nn.Module):
    def __init__(self, num_classes=4, n_meta=8, meta_hidden=64, dropout=0.45):
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
        self.fc = nn.Sequential(
            nn.Linear(256 + meta_hidden, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
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

    def forward(self, sig, meta):
        x = self.pool(self.relu(self.bn(self.conv(sig))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x).view(x.size(0), -1)
        m = self.meta_fc(meta)
        return self.fc(torch.cat((x, m), dim=1))


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


# --- ECG-realistic augmentation (train-time only) ---
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
    wander = amp * torch.sin(2 * np.pi * freq * t + phase)
    return x + wander

def train_augment(sigs, device):
    sigs = aug_time_shift(sigs)
    sigs = aug_baseline_wander(sigs, device)
    sigs = aug_scale_shift(sigs)
    sigs = aug_noise(sigs)
    return sigs

def mixup(sigs, metas, labels, alpha=0.3, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(sigs.size(0), device=device)
    mixed_sigs = lam * sigs + (1 - lam) * sigs[idx]
    mixed_metas = lam * metas + (1 - lam) * metas[idx]
    return mixed_sigs, mixed_metas, labels, labels[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)""")

add_md("## 6. Train the CNN with GroupKFold (5 folds)")
add_code("""CNN_EPOCHS = 32
CNN_PATIENCE = 8
BATCH_SIZE = 256

class_counts = np.bincount(y_train)
alpha = torch.tensor(1.0 / class_counts, dtype=torch.float32)
alpha = (alpha / alpha.sum()).to(DEVICE)

oof_cnn_probs = np.zeros((len(train_df), 4), dtype=np.float32)
cnn_models_state = []
cnn_fold_f1 = []
_cnn_start = time.time()

for fold, (tr_idx, va_idx) in enumerate(FOLD_SPLITS):
    print(f"\\n=== CNN fold {fold} === ({time.time() - _cnn_start:.0f}s elapsed so far)")
    train_ds = ECGDataset(X_sig_train[tr_idx], X_meta_train[tr_idx], y_train[tr_idx])
    val_ds = ECGDataset(X_sig_train[va_idx], X_meta_train[va_idx], y_train[va_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = RobustResNet1D(num_classes=4, n_meta=len(RR_META_COLS)).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS, eta_min=1e-6)

    best_f1, best_state, no_improve = 0.0, None, 0
    for epoch in range(CNN_EPOCHS):
        model.train()
        for sigs, metas, labels in train_loader:
            sigs, metas, labels = sigs.to(DEVICE), metas.to(DEVICE), labels.to(DEVICE)
            sigs = train_augment(sigs, DEVICE)
            sigs, metas, y_a, y_b, lam = mixup(sigs, metas, labels, alpha=0.3, device=DEVICE)
            optimizer.zero_grad()
            outputs = model(sigs, metas)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sigs, metas, labels in val_loader:
                sigs, metas = sigs.to(DEVICE), metas.to(DEVICE)
                preds = torch.argmax(model(sigs, metas), dim=1)
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
            print(f"  epoch {epoch+1}/{CNN_EPOCHS} - val macro F1: {val_f1:.4f} (best: {best_f1:.4f})")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = []
        for sigs, metas, _ in val_loader:
            sigs, metas = sigs.to(DEVICE), metas.to(DEVICE)
            probs.append(torch.softmax(model(sigs, metas), dim=1).cpu().numpy())
        oof_cnn_probs[va_idx] = np.concatenate(probs)

    cnn_fold_f1.append(best_f1)
    cnn_models_state.append(best_state)
    print(f"CNN fold {fold} best macro F1: {best_f1:.4f}")

cnn_oof_f1 = f1_score(y_train, np.argmax(oof_cnn_probs, axis=1), average='macro')
print(f"\\nCNN training took {time.time() - _cnn_start:.0f}s total")
print(f"CNN block-CV OOF macro F1: {cnn_oof_f1:.4f} (fold mean {np.mean(cnn_fold_f1):.4f})")
print(classification_report(y_train, np.argmax(oof_cnn_probs, axis=1), digits=4))""")

add_md("## 7. Train XGBoost on the same folds")
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
sample_weights = compute_sample_weight('balanced', y_train)

for fold, (tr_idx, va_idx) in enumerate(FOLD_SPLITS):
    X_t, y_t = X_stat_train.iloc[tr_idx], y_train[tr_idx]
    X_v, y_v = X_stat_train.iloc[va_idx], y_train[va_idx]
    w_t = sample_weights[tr_idx]

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

add_md("## 8. OOF Ensemble Check")
add_code("""for w_cnn in [0.3, 0.4, 0.5, 0.6, 0.7]:
    blend = w_cnn * oof_cnn_probs + (1 - w_cnn) * oof_xgb_probs
    blend_f1 = f1_score(y_train, np.argmax(blend, axis=1), average='macro')
    print(f"w_cnn={w_cnn:.1f}: block-CV OOF ensemble macro F1 = {blend_f1:.4f}")

# pick the best blend weight found above for the final test-set ensemble
W_CNN = 0.5
print(f"\\nUsing w_cnn={W_CNN} for the final test ensemble (adjust based on the sweep above).")""")

add_md("""## 9. Inference & Submission

We reuse the 5 fold-trained models for both CNN and XGBoost as a bagging ensemble on
`test.csv` (averaging their predictions), rather than a single retrained model -- this
matches the pattern of the prior XGBoost submission and gives a small extra
regularization benefit for the minority classes.""")
add_code("""def predict_cnn_test(states, X_sig, X_meta, device):
    ds = ECGDataset(X_sig, X_meta)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    all_probs = np.zeros((len(X_sig), 4), dtype=np.float32)
    model = RobustResNet1D(num_classes=4, n_meta=X_meta.shape[1]).to(device)
    for state in states:
        model.load_state_dict(state)
        model.eval()
        probs = []
        with torch.no_grad():
            for sigs, metas in loader:
                sigs, metas = sigs.to(device), metas.to(device)
                out_clean = torch.softmax(model(sigs, metas), dim=1)
                out_noisy = torch.softmax(model(aug_noise(sigs, 0.02), metas), dim=1)
                probs.append(((out_clean + out_noisy) / 2).cpu().numpy())
        all_probs += np.concatenate(probs) / len(states)
    return all_probs

print("Running CNN inference on test.csv...")
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
print("\\nIf this run generalizes better than the prior submissions, the predicted")
print("distribution above should sit noticeably closer to train's than the ~85-88% Normal /")
print("~9-14% VEB split both prior models produced.")

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

with open("submission_patient_robust.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Wrote submission_patient_robust.ipynb")
