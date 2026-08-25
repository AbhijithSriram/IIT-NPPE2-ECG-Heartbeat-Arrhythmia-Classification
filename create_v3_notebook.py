import json

cells = []

def _src(text):
    lines = text.split('\n')
    # no trailing newline on the last line -> no phantom blank line in the cell
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _src(text)})

def add_code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": _src(text)})

add_md("""# Recording-Aware ECG Classification with Label-Shift Correction

Two structural properties of this dataset drive the whole approach.

**1. The recording ID is recoverable.** The data description defines
`rr_ratio = pre_rr / (median RR-interval of the recording this beat came from)`. That median
is a per-recording constant, so `pre_rr / rr_ratio` inverts to a fingerprint shared by every
beat of the same recording. Rounded to 3 decimals this gives 71 distinct recordings in
train.csv and 25 in test.csv, essentially disjoint, with very different per-recording label
mixes (one recording is 78.6% Ventricular Ectopic, another 59.3% Supraventricular, another
84.5% Normal). A random row-level split therefore puts beats from the same patient on both
sides, which is why a random-split CV reads ~0.93 while the leaderboard reads ~0.51. Grouping
by recording gives an honest inter-patient CV of ~0.67.

**2. train and test have different class distributions.** Under honest CV the model's
predicted mix matches truth on held-out *training* recordings, but on test.csv the same model
predicts a far more Normal-heavy mix. The cause is visible in the beat counts: train averages
1,011 beats per recording, test 2,757. Train was subsampled to rebalance the classes; test
was left at natural prevalence, where Normal beats dominate at roughly 90%. This is label
shift - `P(x|y)` is stable, `P(y)` is not - so the decision rule, not just the model, has to
be corrected.

Macro F1 weights all four classes equally, so on a ~90% Normal test set plain `argmax` is the
wrong rule. We estimate the test priors and tune per-class log-offsets to maximize macro F1
under the shift, choosing offsets that hold up across several candidate priors rather than
being optimal for one.

No pretrained weights, no transfer learning, no external data.""")

add_md("""## Configuration

`TARGET_PRIOR` is the estimated test-set class prevalence and is the main knob. It is set from
BBSE with the rare classes pulled toward natural prevalence, because BBSE and EM both collapse
ultra-rare classes toward zero - and a Fusion F1 of 0 caps attainable macro F1 at 0.75.""")
add_code("""# Estimated test-set class prevalence: [Normal, SVEB, VEB, Fusion]
TARGET_PRIOR = [0.884, 0.032, 0.078, 0.006]

# 'tuned'       : robust offset search across candidate priors (best on honest CV)
# 'prior_match' : offsets chosen so predicted counts match TARGET_PRIOR
# 'blend'       : average of the two offset vectors
DECISION_MODE = 'tuned'

SEED = 42""")

add_md("## 0. Imports")
add_code("""import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

warnings.filterwarnings('ignore')
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
      f"test {len(test_df)/test_df['rec'].nunique():.0f}")

print("\\nLabel mix within the 10 largest training recordings:")
for r, cnt in train_df['rec'].value_counts().head(10).items():
    d = train_df.loc[train_df['rec'] == r, 'label'].value_counts(normalize=True)
    print(f"  rec={r:.3f} n={cnt:5d}  " + "  ".join(f"c{c}={d.get(c, 0)*100:5.1f}%" for c in range(4)))""")

add_md("""## 2. Feature engineering

Three families of features:

- **Morphology and frequency** on the per-beat z-normalized waveform: QRS width at several
  amplitude thresholds, up/down slopes, P- and T-region areas and energies, FFT magnitudes.
- **Timing**, expressed relative to the recording's own median RR so it carries no
  patient-specific heart-rate scale: `prematurity`, `comp_pause`, `post_over_med`.
- **Patient-adaptive**, and fully unsupervised so it applies identically to test: how far each
  beat deviates from its *own* recording's normal-beat template, and how rare its morphology
  is within that recording. This is what lets ectopic detection transfer across patients - an
  ectopic beat is one that looks unlike that particular patient's usual beat, whatever
  "usual" happens to be for them.

Region boundaries on the 250-sample R-peak-centred window (~1 s at 250 Hz):
P is `[55, 105)`, QRS is `[105, 145)`, T is `[145, 205)`.""")
add_code("""P_SL = slice(55, 105)
QRS_SL = slice(105, 145)
T_SL = slice(145, 205)


def zsig(df):
    s = df[SIG].values.astype(np.float64)
    m = s.mean(1, keepdims=True)
    d = s.std(1, keepdims=True)
    d[d == 0] = 1e-8
    return s, (s - m) / d


def corr_region(A, t, sl):
    \"\"\"Row-wise correlation between each beat and a template, restricted to one region.\"\"\"
    a = A[:, sl]
    b = t[sl]
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean()
    na = np.sqrt((a ** 2).sum(1)) + 1e-8
    nb = np.sqrt((b ** 2).sum()) + 1e-8
    return (a @ b) / (na * nb)


def build_features(sig, sigz, df):
    d1 = np.diff(sig, axis=1)
    d2 = np.diff(d1, axis=1)
    F = {}
    F['mean'] = sig.mean(1)
    F['std'] = sig.std(1)
    F['max'] = sig.max(1)
    F['min'] = sig.min(1)
    F['median'] = np.median(sig, 1)
    F['skew'] = skew(sig, axis=1)
    F['kurt'] = kurtosis(sig, axis=1)
    F['energy'] = (sig ** 2).sum(1)
    F['zc'] = (np.diff(np.sign(sig), axis=1) != 0).sum(1)
    for nm, d in [('d1', d1), ('d2', d2)]:
        F[nm + '_mean'] = d.mean(1)
        F[nm + '_std'] = d.std(1)
        F[nm + '_max'] = d.max(1)
        F[nm + '_min'] = d.min(1)
    fft = np.abs(np.fft.rfft(sigz, axis=1))
    for i in range(1, 11):
        F[f'fft_{i}'] = fft[:, i]

    F['qrs_max'] = sigz[:, 110:140].max(1)
    F['qrs_min'] = sigz[:, 110:140].min(1)
    F['qrs_ptp'] = F['qrs_max'] - F['qrs_min']
    F['qrs_area'] = np.abs(sigz[:, 110:140]).sum(1)
    for th in (0.5, 1.0, 1.5, 2.0):
        F[f'qrs_w{th}'] = (np.abs(sigz[:, 100:150]) > th).sum(1)
    F['pre_area'] = np.abs(sigz[:, P_SL]).sum(1)
    F['post_area'] = np.abs(sigz[:, T_SL]).sum(1)
    F['t_max'] = sigz[:, T_SL].max(1)
    F['t_min'] = sigz[:, T_SL].min(1)
    F['p_max'] = sigz[:, P_SL].max(1)
    F['p_min'] = sigz[:, P_SL].min(1)
    F['p_ptp'] = F['p_max'] - F['p_min']
    F['p_energy'] = (sigz[:, P_SL] ** 2).sum(1)
    F['t_energy'] = (sigz[:, T_SL] ** 2).sum(1)
    F['qrs_energy'] = (sigz[:, QRS_SL] ** 2).sum(1)
    F['p_over_qrs'] = F['p_energy'] / (F['qrs_energy'] + 1e-8)
    F['t_over_qrs'] = F['t_energy'] / (F['qrs_energy'] + 1e-8)
    dz = np.diff(sigz, axis=1)
    F['qrs_upslope'] = dz[:, 105:130].max(1)
    F['qrs_downslope'] = dz[:, 120:145].min(1)

    F['pre_rr'] = df['pre_rr'].values
    F['post_rr'] = df['post_rr'].values
    F['rr_ratio'] = df['rr_ratio'].values
    F['rr_diff'] = F['post_rr'] - F['pre_rr']
    F['post_over_med'] = df['post_rr'].values / np.clip(df['implied_rr'].values, 1e-3, None)
    F['prematurity'] = 1.0 - F['rr_ratio']
    F['comp_pause'] = F['post_over_med'] - 1.0
    F['rr_local_ratio'] = df['pre_rr'].values / np.clip(df['post_rr'].values, 1e-3, None)
    # a premature beat followed by a compensatory pause is the hallmark of an ectopic beat
    F['ectopic_rhythm'] = F['prematurity'] * F['comp_pause']
    return pd.DataFrame(F)


def patient_features(sigz, df, pcs):
    n = len(df)
    keys = ['tpl_corr', 'tpl_l2', 'tpl_linf', 'amp_rel', 'width_rel',
            'rarity_r1', 'rarity_r2', 'nn5_dist', 'nn20_dist', 'dist_centroid',
            'rank_in_rec', 'rec_size', 'corr_dom', 'corr_alt', 'corr_gap',
            'corr_P', 'corr_QRS', 'corr_T', 'qrs_minus_p', 'qrs_minus_t',
            'p_energy_rel', 'qrs_energy_rel', 't_energy_rel',
            'rr_pct_in_rec', 'rr_z_in_rec', 'rec_rr_std',
            'rec_premature_frac', 'rec_wide_frac']
    out = {k: np.zeros(n, dtype=np.float32) for k in keys}

    rr = df['rr_ratio'].values
    recs = df['rec'].values
    prem = 1.0 - rr
    qrs_w = (np.abs(sigz[:, 100:150]) > 1.0).sum(1).astype(float)
    amp = sigz[:, 110:140].max(1) - sigz[:, 110:140].min(1)
    p_en = (sigz[:, P_SL] ** 2).sum(1)
    q_en = (sigz[:, QRS_SL] ** 2).sum(1)
    t_en = (sigz[:, T_SL] ** 2).sum(1)

    for r in np.unique(recs):
        m = np.where(recs == r)[0]
        sub = sigz[m]
        P = pcs[m].astype(np.float32)
        k = len(m)

        onsched = m[(rr[m] > 0.9) & (rr[m] < 1.1)]
        ref = onsched if len(onsched) >= 30 else m
        tpl = np.median(sigz[ref], axis=0)
        tz = (tpl - tpl.mean()) / (tpl.std() + 1e-8)
        out['tpl_corr'][m] = (sub @ tz) / 250.0
        diff = sub - tpl
        out['tpl_l2'][m] = np.sqrt((diff ** 2).mean(1))
        out['tpl_linf'][m] = np.abs(diff).max(1)

        # region-split template correlation: separates SVEB (normal QRS, abnormal P)
        # from VEB (the QRS itself is wide and abnormal)
        cP = corr_region(sub, tpl, P_SL)
        cQ = corr_region(sub, tpl, QRS_SL)
        cT = corr_region(sub, tpl, T_SL)
        out['corr_P'][m] = cP
        out['corr_QRS'][m] = cQ
        out['corr_T'][m] = cT
        out['qrs_minus_p'][m] = cQ - cP
        out['qrs_minus_t'][m] = cQ - cT
        out['p_energy_rel'][m] = p_en[m] / (np.median(p_en[ref]) + 1e-8)
        out['qrs_energy_rel'][m] = q_en[m] / (np.median(q_en[ref]) + 1e-8)
        out['t_energy_rel'][m] = t_en[m] / (np.median(t_en[ref]) + 1e-8)

        ref_amp = np.median(amp[ref])
        ref_w = np.median(qrs_w[ref])
        out['amp_rel'][m] = amp[m] / (ref_amp + 1e-8)
        out['width_rel'][m] = qrs_w[m] / (ref_w + 1e-8)

        rr_m = rr[m]
        out['rr_pct_in_rec'][m] = np.argsort(np.argsort(rr_m)) / max(k - 1, 1)
        out['rr_z_in_rec'][m] = (rr_m - rr_m.mean()) / (rr_m.std() + 1e-8)
        out['rec_rr_std'][m] = rr_m.std()
        out['rec_premature_frac'][m] = (prem[m] > 0.15).mean()
        out['rec_wide_frac'][m] = (qrs_w[m] > 1.5 * ref_w).mean()

        cen = P.mean(0)
        out['dist_centroid'][m] = np.linalg.norm(P - cen, axis=1)
        out['rec_size'][m] = k
        scale = np.median(np.linalg.norm(P - cen, axis=1)) + 1e-8
        r1 = np.zeros(k, np.float32)
        r2 = np.zeros(k, np.float32)
        nn5 = np.zeros(k, np.float32)
        nn20 = np.zeros(k, np.float32)
        for s in range(0, k, 512):
            e = min(s + 512, k)
            d = np.linalg.norm(P[s:e, None, :] - P[None, :, :], axis=2)
            r1[s:e] = (d < 0.5 * scale).sum(1) / k
            r2[s:e] = (d < 1.0 * scale).sum(1) / k
            ds = np.sort(d, axis=1)
            nn5[s:e] = ds[:, min(5, k - 1)]
            nn20[s:e] = ds[:, min(20, k - 1)]
        out['rarity_r1'][m] = r1
        out['rarity_r2'][m] = r2
        out['nn5_dist'][m] = nn5 / scale
        out['nn20_dist'][m] = nn20 / scale
        out['rank_in_rec'][m] = np.argsort(np.argsort(out['tpl_corr'][m])) / max(k - 1, 1)

        if k >= 60:
            km = KMeans(n_clusters=3, n_init=4, random_state=0).fit(P)
            lab = km.labels_
            sizes = np.bincount(lab, minlength=3)
            tps = []
            for ci in np.argsort(sizes)[::-1]:
                t = np.median(sigz[m[lab == ci]], axis=0)
                tps.append((t - t.mean()) / (t.std() + 1e-8))
            c_dom = (sub @ tps[0]) / 250.0
            c_alt = np.maximum((sub @ tps[1]) / 250.0, (sub @ tps[2]) / 250.0)
            out['corr_dom'][m] = c_dom
            out['corr_alt'][m] = c_alt
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

add_md("""## 3. Honest inter-patient CV, averaged over several tree depths

Validation is `GroupKFold` on the recovered recording IDs, so no patient appears on both
sides of a fold. Predictions are averaged over three depth/seed configurations: deep trees can
memorize patient-specific morphology, shallow ones are pushed toward patterns that transfer,
and averaging cuts the variance that matters when only 25 test recordings decide the score.
Each configuration's own OOF score is printed, so it is visible whether shallower generalizes
better here.""")
add_code("""SPLITS = list(GroupKFold(n_splits=5).split(X_tr, y, groups=groups))
for i, (tr, va) in enumerate(SPLITS):
    print(f"  fold {i}: {len(np.unique(groups[tr]))} train recordings / "
          f"{len(np.unique(groups[va]))} held-out recordings")

BASE = dict(objective='multi:softprob', num_class=4, eval_metric='mlogloss',
            subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
            tree_method='hist', device=XGB_DEVICE, n_estimators=700)
CONFIGS = [
    dict(max_depth=6, learning_rate=0.06, random_state=42),
    dict(max_depth=4, learning_rate=0.06, random_state=7),
    dict(max_depth=8, learning_rate=0.04, random_state=2024),
]

t0 = time.time()
oof = np.zeros((len(X_tr), 4))
test_prob = np.zeros((len(X_te), 4))
for cfg in CONFIGS:
    o = np.zeros((len(X_tr), 4))
    t = np.zeros((len(X_te), 4))
    for tr, va in SPLITS:
        clf = xgb.XGBClassifier(**{**BASE, **cfg})
        clf.fit(X_tr.iloc[tr], y[tr],
                sample_weight=compute_sample_weight('balanced', y[tr]), verbose=False)
        o[va] = clf.predict_proba(X_tr.iloc[va])
        t += clf.predict_proba(X_te) / len(SPLITS)
    print(f"  depth={cfg['max_depth']} lr={cfg['learning_rate']} seed={cfg['random_state']}: "
          f"OOF macro F1 = {f1_score(y, o.argmax(1), average='macro'):.4f}")
    oof += o / len(CONFIGS)
    test_prob += t / len(CONFIGS)
print(f"training took {time.time()-t0:.0f}s")

pred_oof = oof.argmax(1)
print(f"\\nHONEST inter-patient OOF macro F1 = {f1_score(y, pred_oof, average='macro'):.4f}")
print(classification_report(y, pred_oof, digits=4))
print("TEST predicted dist (raw argmax):",
      np.round(np.bincount(test_prob.argmax(1), minlength=4) / len(X_te) * 100, 2))""")

add_md("""## 4. Estimate test priors and choose the decision rule

BBSE inverts the OOF confusion matrix against the observed test prediction mix to estimate the
test priors. We keep several candidate priors and pick offsets that do well across all of them,
rather than trusting a single estimate - BBSE is unreliable for a class as rare as Fusion.
`MIN_PER_CLASS` stops the search from abandoning a class outright, since a class F1 of 0
costs a full quarter of macro F1.""")
add_code("""p_train = np.bincount(y, minlength=4) / len(y)
cm = confusion_matrix(y, pred_oof, labels=range(4)).astype(float)
C = cm / cm.sum(1, keepdims=True)
q_test = np.bincount(test_prob.argmax(1), minlength=4) / len(test_prob)
p_bbse, _ = nnls(C.T, q_test)
p_bbse /= p_bbse.sum()

print("train prior     :", np.round(p_train, 4))
print("BBSE estimate   :", np.round(p_bbse, 4))
print("TARGET_PRIOR    :", np.round(np.array(TARGET_PRIOR), 4))
print("implied counts  :", np.round(np.array(TARGET_PRIOR) * len(X_te)).astype(int))

logp_oof = np.log(np.clip(oof, 1e-9, None))
logp_test = np.log(np.clip(test_prob, 1e-9, None))

PRIORS = {'bbse': p_bbse,
          'target': np.array(TARGET_PRIOR),
          'natural': np.array([0.90, 0.025, 0.07, 0.008])}
PRIORS = {k: np.clip(v, 1e-4, None) / np.clip(v, 1e-4, None).sum() for k, v in PRIORS.items()}
W = {k: (p / p_train)[y] for k, p in PRIORS.items()}


def sc(b, k):
    return f1_score(y, (logp_oof + b).argmax(1), average='macro', sample_weight=W[k])


def prior_match_offsets(logp, target, iters=400, lr=0.25):
    target = np.asarray(target, float)
    target = target / target.sum()
    b = np.zeros(4)
    for _ in range(iters):
        cur = np.bincount((logp + b).argmax(1), minlength=4) / len(logp)
        b += lr * (np.log(np.clip(target, 1e-6, None)) - np.log(np.clip(cur, 1e-6, None)))
        if np.max(np.abs(cur - target)) < 2e-4:
            break
    return b


MIN_PER_CLASS = 60
b_match = prior_match_offsets(logp_test, TARGET_PRIOR)

b_tuned = np.zeros(4)
best = float(np.mean([sc(b_tuned, k) for k in PRIORS]))
for _ in range(10):
    improved = False
    for c in range(4):
        for d in (1.2, 0.6, 0.3, 0.15, 0.07, -0.07, -0.15, -0.3, -0.6, -1.2):
            cand = b_tuned.copy()
            cand[c] += d
            if (np.bincount((logp_test + cand).argmax(1), minlength=4) < MIN_PER_CLASS).any():
                continue
            s = float(np.mean([sc(cand, k) for k in PRIORS]))
            if s > best + 1e-6:
                best, b_tuned, improved = s, cand, True
    if not improved:
        break

b = {'prior_match': b_match, 'tuned': b_tuned,
     'blend': 0.5 * (b_match + b_tuned)}[DECISION_MODE]
print(f"\\nUSING '{DECISION_MODE}' -> offsets {np.round(b, 3)}")

print("\\n" + "=" * 72)
print("DIAGNOSTICS FOR ALL DECISION MODES")
print("=" * 72)
for nm, bb in [('argmax', np.zeros(4)), ('prior_match', b_match),
               ('tuned', b_tuned), ('blend', 0.5 * (b_match + b_tuned))]:
    cnts = np.bincount((logp_test + bb).argmax(1), minlength=4)
    scores = {k: sc(bb, k) for k in PRIORS}
    print(f"\\n[{nm}] offsets={np.round(bb, 3)}")
    print(f"  avg={np.mean(list(scores.values())):.4f}  " +
          ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    print(f"  test counts={cnts}  dist={np.round(cnts / cnts.sum() * 100, 2)}")
    for k in PRIORS:
        print(f"    per-class [{k:<8}]",
              np.round(f1_score(y, (logp_oof + bb).argmax(1),
                                average=None, sample_weight=W[k]), 3))""")

add_md("## 5. Predict and submit")
add_code("""final_preds = (logp_test + b).argmax(1)
cnt = np.bincount(final_preds, minlength=4)
print("final test counts      :", cnt)
print("final test distribution:", np.round(cnt / cnt.sum() * 100, 2))
print("TARGET_PRIOR           :", np.round(np.array(TARGET_PRIOR) * 100, 2))

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
with open("submission_v3.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("Wrote submission_v3.ipynb")
