"""
Diagnose the CV-vs-leaderboard gap for the ECG heartbeat arrhythmia competition.

Both prior submissions (DeepECGNet ConvNeXt1D ~0.936 CV, XGBoost ~0.92 CV) scored
only ~0.51 on the public leaderboard. Both used a plain random/stratified
row-level split (train_test_split / StratifiedKFold, stratify=y, random_state=42)
to validate. This script checks whether that split leaks information between
train and validation because many highly-similar beats from the same underlying
ECG recording ("pseudo-patient") are scattered across the (pre-shuffled) train.csv.

It does this in stages:
  1. For every training beat, find its nearest-neighbor beat (by signal shape
     correlation) elsewhere in train.csv, and compare that to a random
     same-class baseline -- to distinguish "these just look alike because
     they're the same arrhythmia class" from "these are near-duplicates".
  2. Do the same nearest-neighbor search from test.csv into train.csv, to see
     whether test beats have close counterparts in train at all.
  3. Cluster near-duplicate beats (signal correlation + matching RR-interval
     context) into pseudo-recording groups via union-find.
  4. Re-run 5-fold CV with a fast XGBoost model two ways on identical
     features: (a) the ORIGINAL random StratifiedKFold methodology (should
     reproduce the inflated ~0.9+ CV), and (b) GroupKFold using the
     pseudo-recording groups (should be a much more honest estimate of
     leaderboard performance).

Everything is logged to diagnostics/<hostname>_<timestamp>_leakage_diagnosis/
so it can be committed and pulled back for review.
"""
import os
import sys
import json
import time
import socket
import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

import torch


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------

def setup_logging(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    logger = logging.getLogger('diagnose_leakage')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(os.path.join(exp_dir, 'diagnosis.log'))
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def resolve_path(explicit, candidates, what):
    tried = [explicit] if explicit else []
    tried += candidates
    for c in tried:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find {what}. Tried: {tried}. Pass the path explicitly.")


def normalize_signals(X):
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    stds[stds == 0] = 1e-8
    return (X - means) / stds


# --------------------------------------------------------------------------
# Fast (vectorized) feature extraction -- same features as train_xgb_features.py
# but without the slow per-row Python loop.
# --------------------------------------------------------------------------

def extract_features_fast(df):
    sig_cols = [f'sig_{i}' for i in range(250)]
    signals = df[sig_cols].values.astype(np.float64)

    d1 = np.diff(signals, axis=1)
    d2 = np.diff(d1, axis=1)

    mean = signals.mean(axis=1)
    std = signals.std(axis=1)
    mx = signals.max(axis=1)
    mn = signals.min(axis=1)
    median = np.median(signals, axis=1)
    sk = skew(signals, axis=1)
    kur = kurtosis(signals, axis=1)
    energy = np.sum(signals ** 2, axis=1)
    zero_crossings = np.sum(np.diff(np.sign(signals), axis=1) != 0, axis=1)

    d1_mean, d1_std = d1.mean(axis=1), d1.std(axis=1)
    d1_max, d1_min = d1.max(axis=1), d1.min(axis=1)
    d2_mean, d2_std = d2.mean(axis=1), d2.std(axis=1)
    d2_max, d2_min = d2.max(axis=1), d2.min(axis=1)

    fft_vals = np.abs(np.fft.rfft(signals, axis=1))
    fft_top = fft_vals[:, 1:11]

    feats = np.column_stack([
        mean, std, mx, mn, median, sk, kur, energy, zero_crossings,
        d1_mean, d1_std, d1_max, d1_min,
        d2_mean, d2_std, d2_max, d2_min,
        fft_top,
    ])
    cols = ['mean', 'std', 'max', 'min', 'median', 'skew', 'kurtosis', 'energy', 'zero_crossings',
            'd1_mean', 'd1_std', 'd1_max', 'd1_min', 'd2_mean', 'd2_std', 'd2_max', 'd2_min'] + \
           [f'fft_{i}' for i in range(1, 11)]
    feat_df = pd.DataFrame(feats, columns=cols)

    for col in ['pre_rr', 'post_rr', 'rr_ratio']:
        df[col] = df[col].fillna(df[col].median())
    feat_df['pre_rr'] = df['pre_rr'].values
    feat_df['post_rr'] = df['post_rr'].values
    feat_df['rr_ratio'] = df['rr_ratio'].values
    feat_df['rr_diff'] = feat_df['post_rr'] - feat_df['pre_rr']
    feat_df['rr_sum'] = feat_df['post_rr'] + feat_df['pre_rr']
    feat_df['rr_norm_pre'] = feat_df['pre_rr'] / (feat_df['rr_sum'] + 1e-6)
    return feat_df


# --------------------------------------------------------------------------
# Union-Find for pseudo-recording clustering
# --------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# --------------------------------------------------------------------------
# Chunked GPU/CPU nearest-neighbor search (cosine correlation on normalized signals)
# --------------------------------------------------------------------------

def nearest_neighbor_search(query, reference, device, chunk_size, logger, self_search, label):
    """For every row of `query`, find its best-matching row in `reference`
    (by dot-product correlation on the pre-normalized 250-sample signal).
    If self_search, `reference is query` and self-matches are excluded."""
    nq = query.shape[0]
    d = query.shape[1]
    Q = torch.tensor(query, dtype=torch.float32, device=device)
    R = Q if self_search else torch.tensor(reference, dtype=torch.float32, device=device)

    nn_idx = np.empty(nq, dtype=np.int64)
    nn_corr = np.empty(nq, dtype=np.float32)

    t0 = time.time()
    n_chunks = (nq + chunk_size - 1) // chunk_size
    for ci, start in enumerate(range(0, nq, chunk_size)):
        end = min(start + chunk_size, nq)
        chunk = Q[start:end]
        sims = (chunk @ R.T) / d
        if self_search:
            rows = torch.arange(start, end, device=device)
            sims[torch.arange(end - start, device=device), rows] = -2.0
        vals, idx = torch.max(sims, dim=1)
        nn_corr[start:end] = vals.cpu().numpy()
        nn_idx[start:end] = idx.cpu().numpy()
        if ci % 5 == 0 or end == nq:
            logger.info(f"  [{label}] {end}/{nq} ({100 * end / nq:.1f}%) - {time.time() - t0:.1f}s elapsed")
    return nn_idx, nn_corr


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-path', default=None)
    parser.add_argument('--test-path', default=None)
    parser.add_argument('--corr-threshold', type=float, default=0.98,
                         help='signal correlation above which a pair is considered a near-duplicate')
    parser.add_argument('--rr-threshold', type=float, default=0.06,
                         help='max |pre_rr| / |post_rr| difference (seconds) for a pair to be grouped')
    parser.add_argument('--n-estimators', type=int, default=400)
    parser.add_argument('--chunk-size', type=int, default=None)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--max-rows', type=int, default=None, help='debug: subsample train rows')
    args = parser.parse_args()

    hostname = socket.gethostname()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"diagnostics/{hostname}_{timestamp}_leakage_diagnosis"
    logger = setup_logging(exp_dir)

    logger.info("=" * 70)
    logger.info("ECG COMPETITION: CV-vs-LEADERBOARD LEAKAGE DIAGNOSIS")
    logger.info("=" * 70)
    logger.info(f"Args: {vars(args)}")

    train_path = resolve_path(
        args.train_path,
        ['dataset/train.csv',
         '../nppe-2-t-2-26-ecg-heartbeat-arrhythmia-classification/nppe2_dataset/train.csv'],
        'train.csv')
    test_path = resolve_path(
        args.test_path,
        ['dataset/test.csv',
         '../nppe-2-t-2-26-ecg-heartbeat-arrhythmia-classification/nppe2_dataset/test.csv'],
        'test.csv')

    logger.info(f"train.csv: {train_path}")
    logger.info(f"test.csv:  {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if args.max_rows:
        train_df = train_df.sample(n=args.max_rows, random_state=42).reset_index(drop=True)
    logger.info(f"train shape: {train_df.shape} | test shape: {test_df.shape}")

    sig_cols = [f'sig_{i}' for i in range(250)]
    sig_train = train_df[sig_cols].values.astype(np.float32)
    sig_test = test_df[sig_cols].values.astype(np.float32)
    y = train_df['label'].values
    ids = train_df['id'].values
    pre_rr = train_df['pre_rr'].fillna(train_df['pre_rr'].median()).values
    post_rr = train_df['post_rr'].fillna(train_df['post_rr'].median()).values

    n = len(train_df)
    sign_train = normalize_signals(sig_train)
    sign_test = normalize_signals(sig_test)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device for nearest-neighbor search: {device}"
                + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))
    chunk_size = args.chunk_size or (4000 if device == 'cuda' else 1000)

    # ---------------------------------------------------------------
    # STEP 1: train-internal nearest neighbor vs random same-class baseline
    # ---------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 1: Train-internal nearest-neighbor correlation")
    logger.info("-" * 70)
    nn_idx, nn_corr = nearest_neighbor_search(
        sign_train, None, device, chunk_size, logger, self_search=True, label="train-internal NN")

    same_label = (y == y[nn_idx])
    logger.info(f"Overall NN correlation: mean={nn_corr.mean():.4f} median={np.median(nn_corr):.4f}")
    logger.info(f"Fraction of NN pairs sharing the same label: {same_label.mean():.4f}")
    for thr in (0.90, 0.95, 0.98, 0.99, 0.995):
        logger.info(f"  fraction of rows with NN corr > {thr}: {(nn_corr > thr).mean():.4f}")

    rng = np.random.default_rng(42)
    logger.info("Random same-class pair correlation baseline (rules out 'they just look alike'):")
    class_baseline = {}
    for cls in sorted(np.unique(y)):
        idx_c = np.where(y == cls)[0]
        if len(idx_c) < 2:
            continue
        i1 = rng.choice(idx_c, size=min(3000, len(idx_c)), replace=True)
        i2 = rng.choice(idx_c, size=min(3000, len(idx_c)), replace=True)
        mask = i1 != i2
        rc = np.sum(sign_train[i1[mask]] * sign_train[i2[mask]], axis=1) / 250.0
        nn_c = nn_corr[idx_c]
        class_baseline[int(cls)] = {
            'random_pair_corr_mean': float(rc.mean()),
            'nn_corr_mean': float(nn_c.mean()),
            'nn_corr_median': float(np.median(nn_c)),
        }
        logger.info(f"  class {cls}: random-pair mean={rc.mean():.4f} | NN mean={nn_c.mean():.4f} "
                    f"median={np.median(nn_c):.4f}  (gap = {nn_c.mean() - rc.mean():.4f})")

    # ---------------------------------------------------------------
    # STEP 2: test-vs-train nearest neighbor
    # ---------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 2: Test-set nearest-neighbor into train (is test 'seen' by train at all?)")
    logger.info("-" * 70)
    test_nn_idx, test_nn_corr = nearest_neighbor_search(
        sign_test, sign_train, device, chunk_size, logger, self_search=False, label="test->train NN")
    logger.info(f"Test->train NN correlation: mean={test_nn_corr.mean():.4f} median={np.median(test_nn_corr):.4f}")
    for thr in (0.90, 0.95, 0.98, 0.99, 0.995):
        logger.info(f"  fraction of TEST rows with a TRAIN neighbor corr > {thr}: {(test_nn_corr > thr).mean():.4f}")
    logger.info("Compare to STEP 1's train-internal fractions above -- if test rows have far fewer/weaker")
    logger.info("near-duplicates in train than train rows have among themselves, that means test beats")
    logger.info("come from sources (patients/recordings) not represented in train at all.")

    # ---------------------------------------------------------------
    # STEP 3: build pseudo-recording groups from train-internal edges
    # ---------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 3: Building pseudo-recording groups (signal match + RR-interval match)")
    logger.info("-" * 70)
    uf = UnionFind(n)
    rr_ok = (np.abs(pre_rr - pre_rr[nn_idx]) < args.rr_threshold) & \
            (np.abs(post_rr - post_rr[nn_idx]) < args.rr_threshold)
    edge_mask = (nn_corr > args.corr_threshold) & rr_ok
    logger.info(f"Edges formed (corr > {args.corr_threshold} AND RR match < {args.rr_threshold}s): "
                f"{edge_mask.sum()} / {n} ({100 * edge_mask.sum() / n:.1f}%)")
    for i in np.where(edge_mask)[0]:
        uf.union(i, int(nn_idx[i]))

    roots = np.array([uf.find(i) for i in range(n)])
    unique_roots, group_sizes = np.unique(roots, return_counts=True)
    n_groups = len(unique_roots)
    logger.info(f"Number of pseudo-groups: {n_groups}")
    logger.info(f"Group size distribution: min={group_sizes.min()} median={np.median(group_sizes):.1f} "
                f"mean={group_sizes.mean():.2f} max={group_sizes.max()}")
    n_singletons = int((group_sizes == 1).sum())
    logger.info(f"Singleton groups (no near-duplicate found): {n_singletons} "
                f"({100 * n_singletons / n_groups:.1f}% of groups, {100 * n_singletons / n:.1f}% of rows)")

    root_to_gid = {r: g for g, r in enumerate(unique_roots)}
    groups = np.array([root_to_gid[r] for r in roots])

    group_df = pd.DataFrame({'group': groups, 'label': y})
    purity_per_group = group_df.groupby('group')['label'].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    logger.info(f"Mean within-group label purity: {purity_per_group.mean():.4f} "
                f"(1.0 = every beat in a pseudo-group shares the same label)")

    pd.DataFrame({
        'id': ids, 'pseudo_group': groups, 'label': y,
        'nn_corr': nn_corr, 'nn_same_label': same_label,
    }).to_csv(os.path.join(exp_dir, 'pseudo_groups.csv'), index=False)

    # ---------------------------------------------------------------
    # STEP 4: feature extraction for the sanity-check model
    # ---------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 4: Extracting features for the XGBoost sanity-check model")
    logger.info("-" * 70)
    t0 = time.time()
    X = extract_features_fast(train_df)
    logger.info(f"Feature extraction done in {time.time() - t0:.1f}s. Shape: {X.shape}")

    import xgboost as xgb
    xgb_device = 'cuda' if device == 'cuda' else 'cpu'
    xgb_params = dict(
        objective='multi:softprob', num_class=4, eval_metric='mlogloss',
        max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', device=xgb_device, random_state=42,
        n_estimators=args.n_estimators,
    )
    logger.info(f"XGBoost params: {xgb_params}")

    def run_cv(split_iter, name):
        oof = np.zeros(n, dtype=np.int64)
        fold_f1 = []
        for fold, (tr_idx, va_idx) in enumerate(split_iter):
            w_tr = compute_sample_weight('balanced', y[tr_idx])
            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=w_tr, verbose=False)
            preds = clf.predict(X.iloc[va_idx])
            oof[va_idx] = preds
            f1 = f1_score(y[va_idx], preds, average='macro')
            fold_f1.append(f1)
            logger.info(f"  [{name}] fold {fold}: macro F1 = {f1:.4f}  (val size={len(va_idx)})")
        overall = f1_score(y, oof, average='macro')
        logger.info(f"[{name}] OOF macro F1 = {overall:.4f}  "
                    f"(fold mean={np.mean(fold_f1):.4f} +/- {np.std(fold_f1):.4f})")
        report = classification_report(y, oof, digits=4, output_dict=True)
        return overall, fold_f1, report

    # ---------------------------------------------------------------
    # STEP 5: reproduce the ORIGINAL (leaky) random CV methodology
    # ---------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 5: Reproducing the ORIGINAL random StratifiedKFold CV (as used in both submissions)")
    logger.info("-" * 70)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    leaky_f1, leaky_fold_f1, leaky_report = run_cv(skf.split(X, y), "RANDOM StratifiedKFold")

    # ---------------------------------------------------------------
    # STEP 6: corrected GroupKFold CV
    # ---------------------------------------------------------------
    logger.info("-" * 70)
    logger.info("STEP 6: Corrected GroupKFold CV (grouped by pseudo-recording)")
    logger.info("-" * 70)
    n_splits_group = min(args.n_splits, n_groups)
    if n_splits_group < 2:
        logger.warning("Not enough pseudo-groups to run GroupKFold; skipping.")
        grouped_f1, grouped_fold_f1, grouped_report = float('nan'), [], {}
    else:
        gkf = GroupKFold(n_splits=n_splits_group)
        grouped_f1, grouped_fold_f1, grouped_report = run_cv(
            gkf.split(X, y, groups=groups), "GROUPED KFold (pseudo-recording-aware)")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Rows: {n} | Pseudo-groups: {n_groups} | Mean group size: {group_sizes.mean():.2f}")
    logger.info(f"RANDOM StratifiedKFold macro F1 (reproduces submitted CV methodology): {leaky_f1:.4f}")
    logger.info(f"GROUPED KFold macro F1 (pseudo-recording-aware, closer to real test):  {grouped_f1:.4f}")
    if not np.isnan(grouped_f1):
        logger.info(f"Gap: {leaky_f1 - grouped_f1:.4f}")
    logger.info("Known Kaggle public LB scores -> deepecgnet: 0.513 | xgb: 0.516")
    logger.info("")

    verdict_leakage = (not np.isnan(grouped_f1)) and (grouped_f1 < leaky_f1 - 0.15)
    if verdict_leakage:
        logger.info("VERDICT: Strong evidence of validation leakage. The random/stratified split lets")
        logger.info("near-duplicate beats from the same underlying ECG recording appear in BOTH the")
        logger.info("train and validation partitions, so local CV massively overstates generalization")
        logger.info("to genuinely unseen recordings/patients -- which is what the Kaggle test set")
        logger.info("contains. From now on, validate with GROUPED KFold (pseudo_group column in")
        logger.info("pseudo_groups.csv), not random/stratified K-fold, before spending a submission.")
    else:
        logger.info("VERDICT: Grouped CV did not collapse as much as expected. Near-duplicate-beat")
        logger.info("leakage may not be the full explanation -- also check label mapping consistency,")
        logger.info("feature/scaler consistency between train and test, and per-class distribution")
        logger.info("shift (see STEP 2 test->train NN stats above).")

    summary = {
        'hostname': hostname,
        'timestamp': timestamp,
        'n_rows': int(n),
        'n_test_rows': int(len(test_df)),
        'n_pseudo_groups': int(n_groups),
        'group_size_mean': float(group_sizes.mean()),
        'group_size_max': int(group_sizes.max()),
        'n_singleton_groups': n_singletons,
        'mean_group_label_purity': float(purity_per_group.mean()),
        'train_internal_nn_corr_mean': float(nn_corr.mean()),
        'train_internal_nn_same_label_frac': float(same_label.mean()),
        'test_to_train_nn_corr_mean': float(test_nn_corr.mean()),
        'class_conditioned_baseline': class_baseline,
        'random_stratkfold_macro_f1': float(leaky_f1),
        'random_stratkfold_fold_f1': [float(v) for v in leaky_fold_f1],
        'random_stratkfold_per_class_f1': {k: v['f1-score'] for k, v in leaky_report.items() if k in ('0', '1', '2', '3')},
        'grouped_kfold_macro_f1': None if np.isnan(grouped_f1) else float(grouped_f1),
        'grouped_kfold_fold_f1': [float(v) for v in grouped_fold_f1],
        'grouped_kfold_per_class_f1': {k: v['f1-score'] for k, v in grouped_report.items() if k in ('0', '1', '2', '3')},
        'known_kaggle_public_lb': {'deepecgnet': 0.513, 'xgb': 0.516},
        'verdict_leakage_confirmed': bool(verdict_leakage),
        'args': vars(args),
    }
    with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved diagnosis.log, summary.json, pseudo_groups.csv to {exp_dir}/")
    logger.info("Commit and push this diagnostics/ folder so it can be pulled and reviewed.")


if __name__ == '__main__':
    main()
