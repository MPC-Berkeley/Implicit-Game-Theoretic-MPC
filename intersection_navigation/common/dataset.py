#!/usr/bin/env python3

import numpy as np
import scipy.linalg as la

from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self, features, targets, feature_mean=None, feature_cov=None, target_mean=None, target_cov=None):
        self.features = features
        self.targets = targets

        self.n = features.shape[1]
        self.d = targets.shape[1]

        if feature_mean is None:
            self.feature_mean = np.mean(self.features, axis=0)
        else:
            self.feature_mean = feature_mean
        if feature_cov is None:
            self.feature_cov = np.cov(self.features, rowvar=False)
        else:
            self.feature_cov = feature_cov

        if target_mean is None:
            self.target_mean = np.mean(self.targets, axis=0)
        else:
            self.target_mean = target_mean
        if target_cov is None:
            self.target_cov = np.cov(self.targets, rowvar=False)
        else:
            self.target_cov = target_cov

        # Regularize the covariance matrices
        reg_value = 1e-6
        self.feature_cov += reg_value * np.eye(self.feature_cov.shape[0])
        self.target_cov += reg_value
        self.features = np.real(la.solve(np.real(la.sqrtm(self.feature_cov)), (self.features - self.feature_mean).T, assume_a='pos').T)
        if self.d > 1:
            self.targets = la.solve(la.sqrtm(self.target_cov), (self.targets - self.target_mean).T, assume_a='pos').T
        else:
            self.targets = (self.targets - self.target_mean)/np.sqrt(self.target_cov)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def get_data_stats(self):
        return self.feature_mean, self.feature_cov, self.target_mean, self.target_cov