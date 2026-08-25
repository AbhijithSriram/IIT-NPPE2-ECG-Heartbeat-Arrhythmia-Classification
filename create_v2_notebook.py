import json

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [l + '\n' for l in text.split('\n')]})

def add_code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + '\n' for l in text.split('\n')]})

add_md("# \U0001F3AF Recording-Aware ECG v2 — P/QRS/T Template Split + Prevalence-Matched Decisions")
add_md("""Builds on the previous submission (public LB **0.611**), which established the two
structural facts about this dataset:

1. **The recording ID is recoverable.** `rr_ratio = pre_rr / median_RR_of_the_recording`, so
   `pre_rr / rr_ratio` inverts to a per-recording constant — **71 recordings in train, 25 in
   test, disjoint**. This gives honest inter-patient `GroupKFold` CV instead of a random split
   that reported a meaningless ~0.93.
2. **train and test have different class distributions.** Train averages 1,011 beats per
   recording, test 2,757: train was subsampled to rebalance classes, test kept its natural
   prevalence (~88-90% Normal). This is label shift — `P(x|y)` stable, `P(y)` very different.

## What changes in v2

The 0.611 run left two clear problems visible in its own output.

**(a) It under-predicted the rare classes.** The offset tuner emitted only **88** Fusion and
**655** SVEB predictions. If the test set really holds ~400 Fusion beats, predicting 88 of them
caps recall at ~16% however good precision is — and macro F1 weights Fusion as a full quarter
of the score. For a rare class, macro F1 is maximized much closer to predicting it at its
*true prevalence*. v2 therefore **matches the predicted distribution to an estimated target
prior** with an explicit floor on the rare classes, instead of letting the tuner suppress them.

**(b) Class 1 (SVEB) lacked the feature that defines it.** An SVEB is a premature beat with an
*abnormal P wave but a normal QRS*. The old template correlation was computed over the whole
250-sample window, which averages that distinction away. v2 computes template correlation
**separately over the P, QRS and T regions**, so "QRS matches this patient's normal beat but
the P wave does not, and the beat came early" becomes directly expressible — which is
precisely the clinical definition of SVEB, and the discriminator that separates it from VEB
(where the QRS itself is wide and abnormal).

Also added: within-recording rhythm-context features (where this beat's RR sits in its own
recording's RR distribution, and how arrhythmic that recording is overall), and a
LightGBM + XGBoost ensemble for variance reduction.

No pretrained weights, no transfer learning, no external data.""")

add_md("""## Configuration

`TARGET_PRIOR` is the estimated test-set class distribution and is the main knob. It is set
from BBSE for classes 0/1/2, with a floor on class 3 because BBSE and EM both collapse
ultra-rare classes toward zero (a known failure mode), and because a Fusion F1 of 0 caps the
attainable macro F1 at 0.75 — below scores already on the leaderboard, so Fusion must be both
present and learnable.""")
add_code("""# ---- main knob: estimated test-set class prevalence ----
# [Normal, SVEB, VEB, Fusion]
TARGET_PRIOR = [0.882, 0.024, 0.088, 0.006]

# 'prior_match' : offsets chosen so predicted counts match TARGET_PRIOR (recommended)
# 'tuned'       : robust offset search across candidate priors (what scored 0.611)
# 'blend'       : average of the two offset vectors
DECISION_MODE = 'prior_match'

USE_LIGHTGBM = True
SEED = 42""")

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

np.random.seed(SEED)
try:
    import torch
    GPU = torch.cuda.is_available()
except Exception:
    GPU = False
XGB_DEVICE = 'cuda' if GPU else 'cpu'
print("XGBoost device:", XGB_DEVICE)""")

add_md("## 1. Load data and recover recording IDs")
add_code("""KAGGLE_INPUT_DIR = '/kaggle/input/nppe-2-t-2-26-ecg-heartbeat-arrhythmia-classification/nppe2_dataset'
TRAIN_PATH = os.path.join(KAGGLE_INPUT_DIR, 'train.csv')
TEST_PATH = os.path.join(KAGGLE_INPUT_DIR, 'test.csv')
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH, TEST_PATH = 'dataset/train.csv', 'dataset/test.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
SIG = [f'sig_{i}' for i in range(250)]

def add_recording_id(df):
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
print(f"Train {train_df.shape} / Test {test_df.shape}")
print(f"Recordings -> train {train_df['rec'].nunique()}, test {test_df['rec'].nunique()}")
print(f"Beats per recording -> train {len(train_df)/train_df['rec'].nunique():.0f}, "
      f"test {len(test_df)/test_df['rec'].nunique():.0f}")""")

add_md("""## 2. Feature engineering

Region boundaries on the 250-sample, R-peak-centred window (~1 s at 250 Hz):
`P ≈ [55,105)`, `QRS ≈ [105,145)`, `T ≈ [145,205)`.""")
add_code("""P_SL = slice(55, 105); QRS_SL = slice(105, 145); T_SL = slice(145, 205)

def zsig(df):
    s = df[SIG].values.astype(np.float64)
    m = s.mean(1, keepdims=True); d = s.std(1, keepdims=True); d[d == 0] = 1e-8
    return s, (s - m) / d

def corr_region(A, t, sl):
    \"\"\"Row-wise correlation between each beat and a template, restricted to one region.\"\"\"
    a = A[:, sl]; b = t[sl]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean()
    na = np.sqrt((a ** 2).sum(1)) + 1e-8
    nb = np.sqrt((b ** 2).sum()) + 1e-8
    return (a @ b) / (na * nb)

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
    F['pre_area'] = np.abs(sigz[P_SL]).sum(1) if False else np.abs(sigz[:, P_SL]).sum(1)
    F['post_area'] = np.abs(sigz[:, T_SL]).sum(1)
    F['t_max'] = sigz[:, T_SL].max(1); F['t_min'] = sigz[:, T_SL].min(1)
    F['p_max'] = sigz[:, P_SL].max(1); F['p_min'] = sigz[:, P_SL].min(1)
    F['p_ptp'] = F['p_max'] - F['p_min']
    F['p_energy'] = (sigz[:, P_SL] ** 2).sum(1)
    F['t_energy'] = (sigz[:, T_SL] ** 2).sum(1)
    F['qrs_energy'] = (sigz[:, QRS_SL] ** 2).sum(1)
    F['p_over_qrs'] = F['p_energy'] / (F['qrs_energy'] + 1e-8)
    F['t_over_qrs'] = F['t_energy'] / (F['qrs_energy'] + 1e-8)
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
    # premature beat followed by a compensatory pause -> hallmark of an ectopic beat
    F['ectopic_rhythm'] = F['prematurity'] * F['comp_pause']
    return pd.DataFrame(F)

def patient_features(sigz, df, pcs):
    \"\"\"All unsupervised -> identical treatment for train and test.\"\"\"
    n = len(df)
    keys = ['tpl_corr','tpl_l2','tpl_linf','amp_rel','width_rel','rarity_r1','rarity_r2',
            'nn5_dist','nn20_dist','dist_centroid','rank_in_rec','rec_size',
            'corr_dom','corr_alt','corr_gap',
            'corr_P','corr_QRS','corr_T','qrs_minus_p','qrs_minus_t',
            'p_energy_rel','qrs_energy_rel','t_energy_rel',
            'rr_pct_in_rec','rr_z_in_rec','rec_rr_std','rec_premature_frac','rec_wide_frac']
    out = {k: np.zeros(n, dtype=np.float32) for k in keys}
    rr = df['rr_ratio'].values; recs = df['rec'].values
    prem = 1.0 - rr
    qrs_w = (np.abs(sigz[:, 100:150]) > 1.0).sum(1).astype(float)
    amp = sigz[:, 110:140].max(1) - sigz[:, 110:140].min(1)
    p_en = (sigz[:, P_SL] ** 2).sum(1)
    q_en = (sigz[:, QRS_SL] ** 2).sum(1)
    t_en = (sigz[:, T_SL] ** 2).sum(1)

    for r in np.unique(recs):
        m = np.where(recs == r)[0]
        sub = sigz[m]; P = pcs[m].astype(np.float32); k = len(m)
        onsched = m[(rr[m] > 0.9) & (rr[m] < 1.1)]
        ref = onsched if len(onsched) >= 30 else m
        tpl = np.median(sigz[ref], axis=0)
        tz = (tpl - tpl.mean()) / (tpl.std() + 1e-8)
        out['tpl_corr'][m] = (sub @ tz) / 250.0
        diff = sub - tpl
        out['tpl_l2'][m] = np.sqrt((diff ** 2).mean(1))
        out['tpl_linf'][m] = np.abs(diff).max(1)

        # --- region-split template correlations (SVEB vs VEB discriminator) ---
        cP = corr_region(sub, tpl, P_SL)
        cQ = corr_region(sub, tpl, QRS_SL)
        cT = corr_region(sub, tpl, T_SL)
        out['corr_P'][m] = cP; out['corr_QRS'][m] = cQ; out['corr_T'][m] = cT
        out['qrs_minus_p'][m] = cQ - cP     # high => normal QRS, abnormal P  => SVEB-like
        out['qrs_minus_t'][m] = cQ - cT
        out['p_energy_rel'][m] = p_en[m] / (np.median(p_en[ref]) + 1e-8)
        out['qrs_energy_rel'][m] = q_en[m] / (np.median(q_en[ref]) + 1e-8)
        out['t_energy_rel'][m] = t_en[m] / (np.median(t_en[ref]) + 1e-8)

        ref_amp = np.median(amp[ref]); ref_w = np.median(qrs_w[ref])
        out['amp_rel'][m] = amp[m] / (ref_amp + 1e-8)
        out['width_rel'][m] = qrs_w[m] / (ref_w + 1e-8)

        # --- rhythm context within this recording ---
        rr_m = rr[m]
        order = np.argsort(np.argsort(rr_m))
        out['rr_pct_in_rec'][m] = order / max(k - 1, 1)
        out['rr_z_in_rec'][m] = (rr_m - rr_m.mean()) / (rr_m.std() + 1e-8)
        out['rec_rr_std'][m] = rr_m.std()
        out['rec_premature_frac'][m] = (prem[m] > 0.15).mean()
        out['rec_wide_frac'][m] = (qrs_w[m] > 1.5 * ref_w).mean()

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
            order2 = np.argsort(sizes)[::-1]
            tps = []
            for ci in order2:
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

add_md("## 3. Honest inter-patient CV (GroupKFold by recording) — XGBoost + LightGBM")
add_code("""SPLITS = list(GroupKFold(n_splits=5).split(X_tr, y, groups=groups))

xgb_params = dict(objective='multi:softprob', num_class=4, eval_metric='mlogloss',
                  max_depth=6, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85,
                  min_child_weight=2, tree_method='hist', device=XGB_DEVICE,
                  random_state=SEED, n_estimators=700)

t0 = time.time()
oof_x = np.zeros((len(X_tr), 4)); test_x = np.zeros((len(X_te), 4))
for i, (tr, va) in enumerate(SPLITS):
    w = compute_sample_weight('balanced', y[tr])
    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X_tr.iloc[tr], y[tr], sample_weight=w, verbose=False)
    oof_x[va] = clf.predict_proba(X_tr.iloc[va])
    test_x += clf.predict_proba(X_te) / len(SPLITS)
    print(f"  XGB fold {i}: macro F1 = {f1_score(y[va], oof_x[va].argmax(1), average='macro'):.4f}")
print(f"XGB done in {time.time()-t0:.0f}s | OOF macro F1 = {f1_score(y, oof_x.argmax(1), average='macro'):.4f}")

oof_l = None
if USE_LIGHTGBM:
    try:
        import lightgbm as lgb
        t0 = time.time()
        oof_l = np.zeros((len(X_tr), 4)); test_l = np.zeros((len(X_te), 4))
        for i, (tr, va) in enumerate(SPLITS):
            w = compute_sample_weight('balanced', y[tr])
            m = lgb.LGBMClassifier(objective='multiclass', num_class=4, n_estimators=600,
                                   learning_rate=0.06, num_leaves=63, subsample=0.85,
                                   subsample_freq=1, colsample_bytree=0.85, min_child_samples=20,
                                   reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(X_tr.iloc[tr], y[tr], sample_weight=w)
            oof_l[va] = m.predict_proba(X_tr.iloc[va])
            test_l += m.predict_proba(X_te) / len(SPLITS)
            print(f"  LGB fold {i}: macro F1 = {f1_score(y[va], oof_l[va].argmax(1), average='macro'):.4f}")
        print(f"LGB done in {time.time()-t0:.0f}s | OOF macro F1 = {f1_score(y, oof_l.argmax(1), average='macro'):.4f}")
    except Exception as e:
        print("LightGBM unavailable/failed, continuing with XGBoost only:", e)
        oof_l = None

if oof_l is not None:
    best_w, best_s = 0.5, -1
    for wgt in np.linspace(0, 1, 11):
        s = f1_score(y, (wgt * oof_x + (1 - wgt) * oof_l).argmax(1), average='macro')
        if s > best_s:
            best_s, best_w = s, wgt
    print(f"\\nbest XGB weight = {best_w:.1f} (blended OOF macro F1 = {best_s:.4f})")
    oof = best_w * oof_x + (1 - best_w) * oof_l
    test_prob = best_w * test_x + (1 - best_w) * test_l
else:
    oof, test_prob = oof_x, test_x

pred_oof = oof.argmax(1)
print(f"\\nHONEST inter-patient OOF macro F1 = {f1_score(y, pred_oof, average='macro'):.4f}")
print(classification_report(y, pred_oof, digits=4))
print("TEST predicted dist (raw argmax):",
      np.round(np.bincount(test_prob.argmax(1), minlength=4)/len(X_te)*100, 2))""")

add_md("""## 4. Estimate test priors and set the decision rule

The previous submission's tuner drove Fusion down to 88 predictions. Because macro F1 treats
Fusion as a full quarter of the score, under-predicting a rare class is far more damaging than
the small precision cost of predicting it at its true rate — so we match the predicted
distribution to `TARGET_PRIOR` instead.""")
add_code("""p_train = np.bincount(y, minlength=4) / len(y)
cm = confusion_matrix(y, pred_oof, labels=range(4)).astype(float)
C = cm / cm.sum(1, keepdims=True)
q_test = np.bincount(test_prob.argmax(1), minlength=4) / len(test_prob)
p_bbse, _ = nnls(C.T, q_test); p_bbse /= p_bbse.sum()
print("train prior     :", np.round(p_train, 4))
print("BBSE estimate   :", np.round(p_bbse, 4))
print("TARGET_PRIOR    :", np.round(np.array(TARGET_PRIOR), 4))
print("implied counts  :", np.round(np.array(TARGET_PRIOR) * len(X_te)).astype(int))

logp_oof = np.log(np.clip(oof, 1e-9, None))
logp_test = np.log(np.clip(test_prob, 1e-9, None))

def prior_match_offsets(logp, target, iters=400, lr=0.25):
    target = np.asarray(target, float); target = target / target.sum()
    b = np.zeros(4)
    for _ in range(iters):
        cur = np.bincount((logp + b).argmax(1), minlength=4) / len(logp)
        b += lr * (np.log(np.clip(target, 1e-6, None)) - np.log(np.clip(cur, 1e-6, None)))
        if np.max(np.abs(cur - target)) < 2e-4:
            break
    return b

PRIORS = {'bbse': p_bbse, 'target': np.array(TARGET_PRIOR),
          'natural': np.array([0.90, 0.025, 0.07, 0.008])}
PRIORS = {k: np.clip(v, 1e-4, None) / np.clip(v, 1e-4, None).sum() for k, v in PRIORS.items()}
W = {k: (p / p_train)[y] for k, p in PRIORS.items()}
def sc(b, k): return f1_score(y, (logp_oof + b).argmax(1), average='macro', sample_weight=W[k])

b_match = prior_match_offsets(logp_test, TARGET_PRIOR)

b_tuned = np.zeros(4); best = float(np.mean([sc(b_tuned, k) for k in PRIORS]))
for _ in range(10):
    improved = False
    for c in range(4):
        for d in (1.2, 0.6, 0.3, 0.15, 0.07, -0.07, -0.15, -0.3, -0.6, -1.2):
            cand = b_tuned.copy(); cand[c] += d
            if (np.bincount((logp_test + cand).argmax(1), minlength=4) < 60).any():
                continue
            s = float(np.mean([sc(cand, k) for k in PRIORS]))
            if s > best + 1e-6:
                best, b_tuned, improved = s, cand, True
    if not improved:
        break

b = {'prior_match': b_match, 'tuned': b_tuned,
     'blend': 0.5 * (b_match + b_tuned)}[DECISION_MODE]

print(f"\\noffsets  prior_match={np.round(b_match,3)}")
print(f"         tuned      ={np.round(b_tuned,3)}")
print(f"USING '{DECISION_MODE}' -> {np.round(b,3)}")
print(f"\\n{'prior':<9} {'argmax':>9} {'match':>9} {'tuned':>9}")
for k in PRIORS:
    print(f"{k:<9} {sc(np.zeros(4),k):>9.4f} {sc(b_match,k):>9.4f} {sc(b_tuned,k):>9.4f}")
print("\\nper-class F1 on honest OOF, reweighted to each prior, using the chosen offsets:")
for k in PRIORS:
    print(f"  {k:<9}", np.round(f1_score(y, (logp_oof + b).argmax(1), average=None, sample_weight=W[k]), 3))""")

add_md("## 5. Predict and submit")
add_code("""final_preds = (logp_test + b).argmax(1)
cnt = np.bincount(final_preds, minlength=4)
print("final test counts      :", cnt)
print("final test distribution:", np.round(cnt / cnt.sum() * 100, 2))
print("TARGET_PRIOR           :", np.round(np.array(TARGET_PRIOR) * 100, 2))
print("\\nPrevious submission (LB 0.611) predicted only 88 Fusion and 655 SVEB beats;")
print("v2 predicts them at roughly their estimated prevalence instead.")

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
with open("submission_v2.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("Wrote submission_v2.ipynb")
