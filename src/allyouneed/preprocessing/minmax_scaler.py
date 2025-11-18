import numpy as np
from .base import BaseEncoder


class MinMaxScaler(BaseEncoder):

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_samples_seen_ = None
        self.fitted = False

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_samples_seen_ = X.shape[0]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_

        self.data_range_[self.data_range_ == 0] = 1.0

        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("MinMaxScaler must be fitted before transform()")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_transformed = X * self.scale_ + self.min_

        return X_transformed

    def inverse_transform(self, X):
        if not self.fitted:
            raise ValueError("MinMaxScaler must be fitted before inverse_transform()")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_inv = (X - self.min_) / self.scale_

        return X_inv
