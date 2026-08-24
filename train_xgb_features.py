import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def extract_features(df, is_test=False):
    print("Extracting features...")
    sig_cols = [f'sig_{i}' for i in range(250)]
    signals = df[sig_cols].values
    
    features = []
    
    # Precompute FFT and derivatives to speed up loops
    diff1 = np.diff(signals, axis=1)
    diff2 = np.diff(diff1, axis=1)
    
    for i in range(len(signals)):
        sig = signals[i]
        d1 = diff1[i]
        d2 = diff2[i]
        
        # 1. Statistical Features
        mean = np.mean(sig)
        std = np.std(sig)
        mx = np.max(sig)
        mn = np.min(sig)
        median = np.median(sig)
        sk = skew(sig)
        kur = kurtosis(sig)
        energy = np.sum(sig ** 2)
        zero_crossings = np.sum(np.diff(np.sign(sig)) != 0)
        
        # 2. Derivative Features (Velocity and Acceleration)
        d1_mean, d1_std, d1_max, d1_min = np.mean(d1), np.std(d1), np.max(d1), np.min(d1)
        d2_mean, d2_std, d2_max, d2_min = np.mean(d2), np.std(d2), np.max(d2), np.min(d2)
        
        # 3. Frequency Domain Features (FFT)
        fft_vals = np.abs(np.fft.rfft(sig))
        # Top 10 frequencies (skip the 0th DC component)
        fft_top = fft_vals[1:11] 
        
        # Collect row features
        row_feats = [
            mean, std, mx, mn, median, sk, kur, energy, zero_crossings,
            d1_mean, d1_std, d1_max, d1_min,
            d2_mean, d2_std, d2_max, d2_min
        ]
        row_feats.extend(fft_top)
        features.append(row_feats)
        
    feat_df = pd.DataFrame(features)
    
    # 4. Enhanced Meta Features
    meta_cols = ['pre_rr', 'post_rr', 'rr_ratio']
    for col in meta_cols:
        df[col] = df[col].fillna(df[col].median())
        
    feat_df['pre_rr'] = df['pre_rr']
    feat_df['post_rr'] = df['post_rr']
    feat_df['rr_ratio'] = df['rr_ratio']
    feat_df['rr_diff'] = df['post_rr'] - df['pre_rr']
    feat_df['rr_sum'] = df['post_rr'] + df['pre_rr']
    feat_df['rr_norm_pre'] = df['pre_rr'] / (feat_df['rr_sum'] + 1e-6)
    
    if is_test:
        return feat_df, df['id'].values
    else:
        return feat_df, df['label'].values

def main():
    TRAIN_PATH = 'dataset/train.csv'
    # Use parent directory path for Linux workstation
    if not os.path.exists(TRAIN_PATH):
        TRAIN_PATH = '../nppe-2-t-2-26-ecg-heartbeat-arrhythmia-classification/nppe2_dataset/train.csv'
        
    train_df = pd.read_csv(TRAIN_PATH)
    
    X, y = extract_features(train_df, is_test=False)
    
    print(f"Feature shape: {X.shape}")
    
    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    # Model configuration
    xgb_params = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'device': 'cuda', # Use GPU on Linux workstation
        'random_state': 42
    }
    
    print("Training XGBoost...")
    models = []
    f1_scores = []
    
    # Class weights for XGBoost (Custom implementation via sample_weight)
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        w_train = sample_weights[train_idx]
        
        clf = xgb.XGBClassifier(**xgb_params, n_estimators=1000)
        
        clf.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        preds = clf.predict(X_val)
        oof_preds[val_idx] = preds
        
        fold_f1 = f1_score(y_val, preds, average='macro')
        f1_scores.append(fold_f1)
        print(f"Fold {fold} Macro F1: {fold_f1:.4f}")
        models.append(clf)
        
    overall_f1 = f1_score(y, oof_preds, average='macro')
    print("-" * 30)
    print(f"CV Macro F1 Scores: {[round(f, 4) for f in f1_scores]}")
    print(f"Overall OOF Macro F1: {overall_f1:.4f}")
    
    # Feature Importance
    importance = models[0].feature_importances_
    top_indices = np.argsort(importance)[::-1][:10]
    print("\nTop 10 Important Features (Fold 0):")
    for idx in top_indices:
        col_name = X.columns[idx]
        print(f"{col_name}: {importance[idx]:.4f}")

    # Save models
    os.makedirs('results_xgb', exist_ok=True)
    for i, model in enumerate(models):
        model.save_model(f'results_xgb/xgb_fold_{i}.json')
    print("Saved models to results_xgb/")

if __name__ == '__main__':
    main()
