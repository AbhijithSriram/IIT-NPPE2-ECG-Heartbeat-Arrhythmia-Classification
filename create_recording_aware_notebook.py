import json

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + '\n' for l in text.split('\n')]})

def add_code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + '\n' for l in text.split('\n')]})

add_md("# \U0001F52C Recording-Aware ECG Classification with Label-Shift Correction")
add_md("""## The two discoveries this notebook is built on

Earlier attempts scored ~0.51-0.55 on the leaderboard while showing 0.92-0.94 on a random
local split. Two findings explain the entire gap.

### 1. The recording ID is recoverable

The data description defines `rr_ratio = pre_rr / (median RR-interval of the recording this
beat came from)`. That median is a **per-recording constant**, so inverting it:

```
implied_median_rr = pre_rr / rr_ratio
```

recovers a fingerprint shared by every beat from the same recording. Rounded to 3 decimals
this yields **71 distinct recordings in train.csv and 25 in test.csv, essentially disjoint**.
The per-recording label mixes are wildly different (one recording is 78.6% Ventricular
Ectopic, another 59.3% Supraventricular, another 84.5% Normal) -- unmistakable patient
structure. A random row-level split therefore puts beats from the *same patient* in both
train and validation, which is why local CV read ~0.93 while the leaderboard read ~0.51.

Using these IDs we get **honest inter-patient `GroupKFold` CV** -- a validation signal that
finally reflects the real task.

### 2. train.csv and test.csv have very different class distributions

With honest CV, the model's predicted distribution on held-out *training* recordings closely
matches the truth (64.9 / 2.9 / 31.8 / 0.4 vs 62.7 / 5.0 / 31.4 / 0.8). But on **test.csv**
the very same model predicts **88.1 / 1.7 / 10.1 / 0.11** -- a completely different mix.

The reason shows up in the beat counts: train averages **1,010 beats per recording** while
test averages **2,757**. Train was *subsampled to rebalance the classes* (Normal beats
discarded, ectopic beats kept); test was left at its natural prevalence, where Normal beats
dominate at ~90%.

So this is a **label-shift** problem: `P(x|y)` is stable but `P(y)` differs sharply between
train and test. We estimate the test priors (BBSE + EM + a ratio-preserving estimate) and
tune per-class decision offsets to maximize *macro* F1 under that shift -- choosing offsets
that are robust across all plausible prior estimates rather than optimal for any single one.

No pretrained weights, no transfer learning, no external data -- everything is derived from
the competition's own CSVs.""")

add_md("## 0. Imports")
add_code("""import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from scipy.optimize import nnls
import xgboost as xgb

SEED = 42
np.random.seed(SEED)
try:
    import torch
    GPU = torch.cuda.is_available()
except Exception:
    GPU = False
XGB_DEVICE = 'cuda' if GPU else 'cpu'
print("XGBoost device:", XGB_DEVICE)""")

add_md("## 1. Load data")
add_code("""KAGGLE_INPUT_DIR = '/kaggle/input/nppe-2-t-2-26-ecg-heartbeat-arrhythmia-classification/nppe2_dataset'
TRAIN_PATH = os.path.join(KAGGLE_INPUT_DIR, 'train.csv')
TEST_PATH = os.path.join(KAGGLE_INPUT_DIR, 'test.csv')
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH, TEST_PATH = 'dataset/train.csv', 'dataset/test.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Train {train_df.shape}, Test {test_df.shape}")
SIG = [f'sig_{i}' for i in range(250)]""")

add_md("""## 2. Recover the recording ID

`pre_rr / rr_ratio` reconstructs each recording's median RR-interval -- a constant shared by
all beats from that recording.""")
add_code("""def add_recording_id(df):
    df = df.copy()
    pre = df['pre_rr'].astype(float).values
    ratio = df['rr_ratio'].astype(float).values
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = pre / ratio
    imp[~np.isfinite(imp)] = np.nan
    imp[np.isnan(imp)] = np.nanmedian(imp)
    df['implied_rr'] = imp
    df['rec'] = np.round(imp, 3)
    for c in ['pre_rr', 'post_rr', 'rr_ratio']:
        df[c] = df[c].fillna(df[c].median())
    return df

train_df = add_recording_id(train_df)
test_df = add_recording_id(test_df)

print(f"Recordings recovered -> train: {train_df['rec'].nunique()}, test: {test_df['rec'].nunique()}")
print(f"Mean beats per recording -> train: {len(train_df)/train_df['rec'].nunique():.0f}, "
      f"test: {len(test_df)/test_df['rec'].nunique():.0f}")
shared = set(train_df['rec']) & set(test_df['rec'])
print(f"Fingerprints appearing in both: {len(shared)} (coincidental median-RR collisions)")

print("\\nLabel mix within the 10 largest training recordings:")
top = train_df['rec'].value_counts().head(10)
for r, cnt in top.items():
    d = train_df.loc[train_df['rec'] == r, 'label'].value_counts(normalize=True)
    print(f"  rec={r:.3f} n={cnt:5d}  " + "  ".join(f"c{c}={d.get(c,0)*100:5.1f}%" for c in range(4)))""")

add_md("## 3. Feature engineering")
add_md("""Three families:

- **Morphology / frequency** on the per-beat z-normalized waveform (QRS width at several
  amplitude thresholds, up/down slopes, P- and T-region areas, FFT magnitudes).
- **Timing**, expressed relative to the recording's own median RR so it carries no
  patient-specific heart-rate scale (`prematurity`, `comp_pause`, `post_over_med`).
- **Patient-adaptive** (fully unsupervised, so it applies identically to test): every beat is
  described by how much it deviates from *its own recording's* normal-beat template, and by
  how **rare** its morphology is within that recording. This is what makes ectopic detection
  transfer across patients -- an ectopic beat is one that looks unlike that particular
  patient's usual beat, regardless of what "usual" happens to look like for them.""")
add_code("""def zsig(df):
    s = df[SIG].values.astype(np.float64)
    m = s.mean(1, keepdims=True); d = s.std(1, keepdims=True); d[d == 0] = 1e-8
    return s, (s - m) / d

def build_features(sig, sigz, df):
    d1 = np.diff(sig, axis=1); d2 = np.diff(d1, axis=1)
    F = {}
    F['mean'] = sig.mean(1); F['std'] = sig.std(1)
    F['max'] = sig.max(1); F['min'] = sig.min(1)
    F['median'] = np.median(sig, 1)
    F['skew'] = skew(sig, axis=1); F['kurt'] = kurtosis(sig, axis=1)
    F['energy'] = (sig ** 2).sum(1)
    F['zc'] = (np.diff(np.sign(sig), axis=1) != 0).sum(1)
    for nm, d in [('d1', d1), ('d2', d2)]:
        F[nm + '_mean'] = d.mean(1); F[nm + '_std'] = d.std(1)
        F[nm + '_max'] = d.max(1); F[nm + '_min'] = d.min(1)
    fft = np.abs(np.fft.rfft(sigz, axis=1))
    for i in range(1, 11):
        F[f'fft_{i}'] = fft[:, i]
    F['qrs_max'] = sigz[:, 110:140].max(1)
    F['qrs_min'] = sigz[:, 110:140].min(1)
    F['qrs_ptp'] = F['qrs_max'] - F['qrs_min']
    F['qrs_area'] = np.abs(sigz[:, 110:140]).sum(1)
    for th in (0.5, 1.0, 1.5, 2.0):
        F[f'qrs_w{th}'] = (np.abs(sigz[:, 100:150]) > th).sum(1)
    F['pre_area'] = np.abs(sigz[:, 60:110]).sum(1)
    F['post_area'] = np.abs(sigz[:, 140:200]).sum(1)
    F['t_max'] = sigz[:, 145:200].max(1); F['t_min'] = sigz[:, 145:200].min(1)
    F['p_max'] = sigz[:, 60:105].max(1)
    dz = np.diff(sigz, axis=1)
    F['qrs_upslope'] = dz[:, 105:130].max(1)
    F['qrs_downslope'] = dz[:, 120:145].min(1)
    F['pre_rr'] = df['pre_rr'].values; F['post_rr'] = df['post_rr'].values
    F['rr_ratio'] = df['rr_ratio'].values
    F['rr_diff'] = F['post_rr'] - F['pre_rr']
    F['post_over_med'] = df['post_rr'].values / np.clip(df['implied_rr'].values, 1e-3, None)
    F['prematurity'] = 1.0 - F['rr_ratio']
    F['comp_pause'] = F['post_over_med'] - 1.0
    F['rr_local_ratio'] = df['pre_rr'].values / np.clip(df['post_rr'].values, 1e-3, None)
    return pd.DataFrame(F)

def patient_features(sigz, df, pcs):
    n = len(df)
    keys = ['tpl_corr','tpl_l2','tpl_linf','amp_rel','width_rel','rarity_r1','rarity_r2',
            'nn5_dist','nn20_dist','dist_centroid','rank_in_rec','rec_size',
            'corr_dom','corr_alt','corr_gap']
    out = {k: np.zeros(n, dtype=np.float32) for k in keys}
    rr = df['rr_ratio'].values; recs = df['rec'].values
    qrs_w = (np.abs(sigz[:, 100:150]) > 1.0).sum(1).astype(float)
    amp = sigz[:, 110:140].max(1) - sigz[:, 110:140].min(1)
    for r in np.unique(recs):
        m = np.where(recs == r)[0]
        sub = sigz[m]; P = pcs[m].astype(np.float32); k = len(m)
        onsched = m[(rr[m] > 0.9) & (rr[m] < 1.1)]
        tpl = np.median(sigz[onsched], axis=0) if len(onsched) >= 30 else np.median(sub, axis=0)
        tz = (tpl - tpl.mean()) / (tpl.std() + 1e-8)
        out['tpl_corr'][m] = (sub @ tz) / 250.0
        diff = sub - tpl
        out['tpl_l2'][m] = np.sqrt((diff ** 2).mean(1))
        out['tpl_linf'][m] = np.abs(diff).max(1)
        ref_amp = np.median(amp[onsched]) if len(onsched) >= 30 else np.median(amp[m])
        out['amp_rel'][m] = amp[m] / (ref_amp + 1e-8)
        ref_w = np.median(qrs_w[onsched]) if len(onsched) >= 30 else np.median(qrs_w[m])
        out['width_rel'][m] = qrs_w[m] / (ref_w + 1e-8)
        cen = P.mean(0)
        out['dist_centroid'][m] = np.linalg.norm(P - cen, axis=1)
        out['rec_size'][m] = k
        scale = np.median(np.linalg.norm(P - cen, axis=1)) + 1e-8
        r1 = np.zeros(k, np.float32); r2 = np.zeros(k, np.float32)
        nn5 = np.zeros(k, np.float32); nn20 = np.zeros(k, np.float32)
        CH = 512
        for s in range(0, k, CH):
            e = min(s + CH, k)
            d = np.linalg.norm(P[s:e, None, :] - P[None, :, :], axis=2)
            r1[s:e] = (d < 0.5 * scale).sum(1) / k
            r2[s:e] = (d < 1.0 * scale).sum(1) / k
            ds = np.sort(d, axis=1)
            nn5[s:e] = ds[:, min(5, k - 1)]; nn20[s:e] = ds[:, min(20, k - 1)]
        out['rarity_r1'][m] = r1; out['rarity_r2'][m] = r2
        out['nn5_dist'][m] = nn5 / scale; out['nn20_dist'][m] = nn20 / scale
        out['rank_in_rec'][m] = np.argsort(np.argsort(out['tpl_corr'][m])) / max(k - 1, 1)
        if k >= 60:
            km = KMeans(n_clusters=3, n_init=4, random_state=0).fit(P)
            lab = km.labels_; sizes = np.bincount(lab, minlength=3)
            order = np.argsort(sizes)[::-1]
            tps = []
            for ci in order:
                t = np.median(sigz[m[lab == ci]], axis=0)
                tps.append((t - t.mean()) / (t.std() + 1e-8))
            c_dom = (sub @ tps[0]) / 250.0
            c_alt = np.maximum((sub @ tps[1]) / 250.0, (sub @ tps[2]) / 250.0)
            out['corr_dom'][m] = c_dom; out['corr_alt'][m] = c_alt
            out['corr_gap'][m] = c_dom - c_alt
        else:
            out['corr_dom'][m] = out['tpl_corr'][m]
            out['corr_alt'][m] = out['tpl_corr'][m]
    return pd.DataFrame(out)

t0 = time.time()
sig_tr, sigz_tr = zsig(train_df)
sig_te, sigz_te = zsig(test_df)
Xb_tr = build_features(sig_tr, sigz_tr, train_df)
Xb_te = build_features(sig_te, sigz_te, test_df)
pca = PCA(n_components=20, random_state=SEED).fit(sigz_tr)
Xp_tr = patient_features(sigz_tr, train_df, pca.transform(sigz_tr))
Xp_te = patient_features(sigz_te, test_df, pca.transform(sigz_te))
X_tr = pd.concat([Xb_tr, Xp_tr], axis=1)
X_te = pd.concat([Xb_te, Xp_te], axis=1)
y = train_df['label'].values
groups = train_df['rec'].values
print(f"Features built in {time.time()-t0:.0f}s -> train {X_tr.shape}, test {X_te.shape}")""")

add_md("## 4. Honest inter-patient CV (GroupKFold by recording)")
add_code("""SPLITS = list(GroupKFold(n_splits=5).split(X_tr, y, groups=groups))
for i, (tr, va) in enumerate(SPLITS):
    print(f"  fold {i}: {len(np.unique(groups[tr]))} train recordings / "
          f"{len(np.unique(groups[va]))} held-out recordings")

params = dict(objective='multi:softprob', num_class=4, eval_metric='mlogloss',
              max_depth=6, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85,
              min_child_weight=2, tree_method='hist', device=XGB_DEVICE,
              random_state=SEED, n_estimators=700)

t0 = time.time()
oof = np.zeros((len(X_tr), 4)); test_prob = np.zeros((len(X_te), 4))
for i, (tr, va) in enumerate(SPLITS):
    w = compute_sample_weight('balanced', y[tr])
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr.iloc[tr], y[tr], sample_weight=w, verbose=False)
    oof[va] = clf.predict_proba(X_tr.iloc[va])
    test_prob += clf.predict_proba(X_te) / len(SPLITS)
    print(f"  fold {i} macro F1 = {f1_score(y[va], oof[va].argmax(1), average='macro'):.4f}")
print(f"training took {time.time()-t0:.0f}s")

pred_oof = oof.argmax(1)
print(f"\\nHONEST inter-patient OOF macro F1 = {f1_score(y, pred_oof, average='macro'):.4f}")
print(classification_report(y, pred_oof, digits=4))
print("OOF predicted dist :", np.round(np.bincount(pred_oof, minlength=4)/len(y)*100, 2))
print("train true dist    :", np.round(np.bincount(y)/len(y)*100, 2))
print("TEST predicted dist:", np.round(np.bincount(test_prob.argmax(1), minlength=4)/len(X_te)*100, 2))
print("\\n^ the test distribution differs sharply from train -- this is the label shift.")""")

add_md("""## 5. Estimate the test-set class priors

Three independent estimates:
- **BBSE** inverts the OOF confusion matrix against the observed test prediction mix.
- **EM** (Saerens-Latinne-Decaestecker) iteratively re-estimates priors from the test posteriors.
- **Ratio-preserved**: since train kept the ectopic beats whole and only subsampled Normal,
  the ratios *among* ectopic classes should survive; we rescale them against the estimated
  test VEB rate.

BBSE and EM are unreliable for a class as rare as Fusion (they drive it toward zero), so we
keep several candidate priors and optimize for robustness across all of them rather than
trusting any single one.""")
add_code("""p_train = np.bincount(y, minlength=4) / len(y)
cm = confusion_matrix(y, pred_oof, labels=range(4)).astype(float)
C = cm / cm.sum(1, keepdims=True)
q_test = np.bincount(test_prob.argmax(1), minlength=4) / len(test_prob)

p_bbse, _ = nnls(C.T, q_test); p_bbse /= p_bbse.sum()

p_src = np.full(4, 0.25)   # model trained with 'balanced' weights -> ~uniform implicit prior
p_em = q_test.copy()
for _ in range(300):
    post = test_prob * (p_em / p_src)
    post /= post.sum(1, keepdims=True)
    new = post.mean(0)
    if np.max(np.abs(new - p_em)) < 1e-9:
        p_em = new; break
    p_em = new

V = float(np.clip(p_bbse[2], 0.02, 0.5))
p_ratio = np.array([0.0, p_train[1]/p_train[2]*V, V, p_train[3]/p_train[2]*V])
p_ratio[0] = 1.0 - p_ratio[1:].sum()

print("train prior      :", np.round(p_train, 4))
print("BBSE  estimate   :", np.round(p_bbse, 4))
print("EM    estimate   :", np.round(p_em, 4))
print("ratio-preserved  :", np.round(p_ratio, 4))

# EM collapses class 1 and class 3 to ~0 here. A prior that declares a class absent makes
# that class's F1 meaningless to optimize against, and including it drags the robust
# objective toward abandoning classes. We report it but exclude it from the tuning set.
print("\\n(EM is excluded from tuning below: it collapses rare classes to ~0, which is a")
print(" known failure mode of EM prior-estimation when a class is very rare.)")

PRIORS = {
    'bbse': p_bbse,
    'ratio': p_ratio,
    'natural': np.array([0.90, 0.025, 0.07, 0.008]),
}
PRIORS = {k: np.clip(v, 1e-4, None) / np.clip(v, 1e-4, None).sum() for k, v in PRIORS.items()}
PRIORS['mid'] = np.mean(list(PRIORS.values()), axis=0)
PRIORS['mid'] /= PRIORS['mid'].sum()
for k, v in PRIORS.items():
    print(f"  {k:<9}", np.round(v, 4))""")

add_md("""## 6. Tune decision offsets for macro F1 under the shift

Macro F1 weights all four classes equally, so on a test set that is ~90% Normal the optimal
decision rule is *not* plain `argmax`. We reweight the honest OOF predictions to look like
each candidate test prior, then coordinate-ascent per-class log-offsets to maximize the
average macro F1 **across all candidate priors** -- with a constraint that no class is ever
abandoned entirely (abandoning a class that is actually present forces its F1 to 0 and costs
a full quarter of the metric).""")
add_code("""logp_oof = np.log(np.clip(oof, 1e-9, None))
logp_test = np.log(np.clip(test_prob, 1e-9, None))
W = {k: (p / p_train)[y] for k, p in PRIORS.items()}

def score(b, k):
    return f1_score(y, (logp_oof + b).argmax(1), average='macro', sample_weight=W[k])

def avg_score(b):
    return float(np.mean([score(b, k) for k in PRIORS]))

MIN_PER_CLASS = 60
b = np.zeros(4); best = avg_score(b)
print(f"average shift-weighted macro F1 @ plain argmax = {best:.4f}")
for _ in range(10):
    improved = False
    for c in range(4):
        for d in (1.2, 0.6, 0.3, 0.15, 0.07, -0.07, -0.15, -0.3, -0.6, -1.2):
            cand = b.copy(); cand[c] += d
            if (np.bincount((logp_test + cand).argmax(1), minlength=4) < MIN_PER_CLASS).any():
                continue
            s = avg_score(cand)
            if s > best + 1e-6:
                best, b, improved = s, cand, True
    if not improved:
        break

print(f"average shift-weighted macro F1 @ tuned offsets = {best:.4f}")
print("offsets:", np.round(b, 3))
print(f"\\n{'prior':<10} {'argmax':>9} {'tuned':>9}")
for k in PRIORS:
    print(f"{k:<10} {score(np.zeros(4), k):>9.4f} {score(b, k):>9.4f}")
print("\\nper-class F1 under each prior (tuned):")
for k in PRIORS:
    print(f"  {k:<10}", np.round(f1_score(y, (logp_oof + b).argmax(1),
                                          average=None, sample_weight=W[k]), 3))""")

add_md("## 7. Predict and submit")
add_code("""final_preds = (logp_test + b).argmax(1)

print("Final test distribution:", np.round(np.bincount(final_preds, minlength=4)/len(final_preds)*100, 2))
print("Final test counts      :", np.bincount(final_preds, minlength=4))
print("Estimated test priors  :", np.round(PRIORS['mid'] * 100, 2))
print("\\n(For reference, earlier ~0.51-0.55 submissions predicted 85-88% Normal with no")
print(" shift correction and were tuned against a random-split CV of ~0.93.)")

submission = pd.DataFrame({'id': test_df['id'].values, 'label': final_preds})
submission.to_csv('submission.csv', index=False)
print("\\nSaved submission.csv", submission.shape)
submission.head()""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
with open("submission_recording_aware.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("Wrote submission_recording_aware.ipynb")
