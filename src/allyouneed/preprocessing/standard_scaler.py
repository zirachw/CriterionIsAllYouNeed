import numpy as np
from .base import BaseEncoder


class StandardScaler(BaseEncoder):

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        self.n_samples_seen_ = None
        self.fitted = False

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_samples_seen_ = X.shape[0]

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = None

        if self.with_std:
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
        else:
            self.scale_ = None

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("StandardScaler must be fitted before transform()")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_transformed = X.copy()

        if self.with_mean:
            X_transformed = X_transformed - self.mean_

        if self.with_std:
            X_transformed = X_transformed / self.scale_

        return X_transformed

    def inverse_transform(self, X):
        if not self.fitted:
            raise ValueError("StandardScaler must be fitted before inverse_transform()")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_inv = X.copy()

        if self.with_std:
            X_inv = X_inv * self.scale_

        if self.with_mean:
            X_inv = X_inv + self.mean_

        return X_inv
