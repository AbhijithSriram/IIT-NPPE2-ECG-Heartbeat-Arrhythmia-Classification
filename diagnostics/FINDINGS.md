# CV-vs-Leaderboard Gap: Findings (2026-08-24)

Both submissions used a plain random/stratified row-level split
(`train_test_split`/`StratifiedKFold(shuffle=True, random_state=42)`) to
validate, and both got a great "CV" (deepecgnet ~0.936, xgb ~0.92) but a
public LB of only 0.513 / 0.516. Full run: `diagnose_leakage.py`, log/summary
in `Abhijith_Sriram_20260824_105146_leakage_diagnosis/`.

## Ruled out

1. **Literal near-duplicate-beat leakage across the random split.**
   Train-internal nearest-neighbor signal correlation is far above the
   random same-class baseline (e.g. class 2: NN corr median 0.993 vs random
   same-class pairs mean 0.19), so near-duplicate beats genuinely exist in
   train.csv. But clustering them into pseudo-recording groups (signal
   corr > 0.98 AND matching RR-interval context) and re-running CV with
   `GroupKFold` on those groups barely moved the score: **0.9139 (random)
   vs 0.9106 (grouped)**. If this were the dominant leakage mechanism, the
   grouped number should have collapsed toward 0.51-0.52. It didn't &mdash;
   the pseudo-groups found this way are small (mean size 1.5, 53% of rows
   in a group of >1), too narrow to represent real patient/recording
   identity.

2. **Simple prior/calibration mismatch.** Validated the recalibration
   trick (`recalibrate.py`) on a genuinely matched-distribution held-out
   split (where train prior == true val prior): it only recovered
   ~0.005 macro F1. Not remotely enough to explain a 0.40 gap, and when
   applied to the real test predictions it barely shifted the aggregate
   distribution and collapsed class 3 (Fusion) toward zero &mdash; a sign
   the raw per-example confidence for that class is already too flat to
   fix by rescaling. **Do not submit a recalibrated file as-is.**

## Strong, confirmed evidence: inter-patient generalization failure

Ran inference with the ACTUAL trained model artifacts already in this repo
(`results_xgb/xgb_fold_*.json`, `results/..._Monster_SE_ConvNeXt1D/best_model.pth`)
on the real `test.csv`, and compared predicted class distribution against
train's true label distribution:

| Class | Train share | XGB predicted | DeepECGNet predicted |
|---|---|---|---|
| 0 Normal | 62.7% | 88.2% | 85.3% |
| 1 SVEB | 5.0% | 1.1% | 0.5% |
| 2 VEB | 31.4% | **9.2%** | **14.0%** |
| 3 Fusion | 0.8% | 1.6% | 0.2% |

Both models &mdash; despite being architecturally very different (raw-signal
ConvNeXt1D CNN vs. hand-engineered statistical/FFT features into XGBoost)
&mdash; collapse toward predicting "Normal" and under-predict Ventricular
Ectopic by roughly the same ~15-17k-beat margin. This is the textbook
inter-patient generalization failure documented in ECG arrhythmia
literature (AAMI inter-patient paradigm): a model trained and validated
intra-patient effectively memorizes per-patient QRS/ectopic-beat
morphology, and that does not transfer to a genuinely new patient's beat
shape, so on unseen recordings it defaults to the majority class. It
happening almost identically for two very different model families is
what rules out a model-specific bug and points at the data/validation
setup + the inherent difficulty of the task.

The XGB notebook's own header comment ("this approach avoids Patient
Leakage by extracting statistical...features instead of memorizing raw
morphologies") turned out to be a false premise &mdash; aggregate stats
(skew, kurtosis, FFT peaks, etc.) still encode enough per-patient
physiological signature for a 1000-tree ensemble to latch onto, so it
failed on new patients almost exactly like the raw-signal model did.

## Files in this folder

- `repro_xgb_submission.csv`, `repro_deepecgnet_submission.csv` &mdash;
  predictions reproduced from the actual saved model artifacts, to confirm
  they match what was likely submitted (sanity check, not new).
- `ensemble_submission.csv` &mdash; simple 50/50 average of both models'
  softmax probabilities. Zero new training. Lowest-risk candidate for the
  next submission: two structurally different models agree on 89.95% of
  test beats, so there's a real 10% disagreement zone ensembling can
  arbitrate.
- `*_recalibrated_submission.csv` &mdash; exploratory only, NOT
  recommended as-is (see "ruled out" above).
- `xgb_test_probs.npy`, `deepecgnet_test_probs.npy` &mdash; raw softmax
  probabilities on test.csv from both models, for further ensembling/
  analysis.
- `pseudo_groups.csv` (in the `..._leakage_diagnosis/` subfolder) &mdash;
  id -> pseudo-recording-group mapping, for anyone who wants to build on
  the grouped-CV idea despite its limited signal here.

## Recommendation for the remaining 3 submissions

1. Try `ensemble_submission.csv` first &mdash; safest, ready now.
2. If time allows, retrain specifically targeting inter-patient
   generalization for class 2 (VEB) and class 1 (SVEB): lean harder on
   RR-interval/timing features (physiologically patient-invariant, unlike
   raw QRS shape), add stronger regularization / simpler models
   (paradoxically often generalizes better across patients), and use
   RR-interval-aware augmentation (small time-warp/jitter, baseline
   wander) rather than trusting local CV of any kind &mdash; local CV on
   this dataset is not a reliable proxy for LB, random OR
   pseudo-grouped.
3. Keep the current best real LB submission (xgb, 0.516) as the
   fallback/final choice if further attempts don't clearly beat it.
