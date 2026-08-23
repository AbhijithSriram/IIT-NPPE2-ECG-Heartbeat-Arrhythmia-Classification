import os
import socket
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def setup_experiment(model_name):
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"results/{hostname}_{timestamp}_{model_name}"
    os.makedirs(exp_dir, exist_ok=True)

    # Setup logging
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(exp_dir, 'training.log'))
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return exp_dir, logger

class ECGDataset(Dataset):
    def __init__(self, signals, features, labels=None):
        self.signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.signals[idx], self.features[idx], self.labels[idx]
        return self.signals[idx], self.features[idx]

def load_data(train_path, test_path=None):
    print(f"Loading data from {train_path}...")
    train_df = pd.read_csv(train_path)

    sig_cols = [f'sig_{i}' for i in range(250)]
    time_cols = ['pre_rr', 'post_rr', 'rr_ratio']

    X_sig_train = train_df[sig_cols].values
    X_time_train = train_df[time_cols].values
    y_train = train_df['label'].values

    imputer = SimpleImputer(strategy='median')
    X_time_train = imputer.fit_transform(X_time_train)

    scaler = StandardScaler()
    X_time_train = scaler.fit_transform(X_time_train)

    def normalize_signals(X):
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds[stds == 0] = 1e-8
        return (X - means) / stds

    X_sig_train = normalize_signals(X_sig_train)

    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        X_sig_test = test_df[sig_cols].values
        X_time_test = test_df[time_cols].values
        X_time_test = imputer.transform(X_time_test)
        X_time_test = scaler.transform(X_time_test)
        X_sig_test = normalize_signals(X_sig_test)
        return X_sig_train, X_time_train, y_train, X_sig_test, X_time_test

    return X_sig_train, X_time_train, y_train

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()
