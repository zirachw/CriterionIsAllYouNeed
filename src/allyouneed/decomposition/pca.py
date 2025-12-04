import numpy as np
from .base import BaseDecomposition


class PCA(BaseDecomposition):

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_samples_ = None
        self.n_features_ = None
        self.fitted = False

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_samples_, self.n_features_ = X.shape

        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        components = Vt
        explained_variance = (S ** 2) / (self.n_samples_ - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance

        if self.n_components is None:
            n_components = min(self.n_samples_, self.n_features_)
        elif isinstance(self.n_components, (int, np.integer)):
            if self.n_components > len(S):
                raise ValueError(
                    f"n_components={self.n_components} is too large. "
                    f"Maximum is {len(S)}"
                )
            n_components = self.n_components
        elif isinstance(self.n_components, (float, np.floating)):
            if not 0 < self.n_components < 1:
                raise ValueError(
                    f"n_components={self.n_components} must be between 0 and 1 "
                    "when specified as a float"
                )
            cumsum = np.cumsum(explained_variance_ratio)
            n_components = np.searchsorted(cumsum, self.n_components) + 1
            n_components = min(n_components, len(S))
        else:
            raise ValueError(
                f"n_components must be int, float, or None, got {type(self.n_components)}"
            )

        self.n_components_ = n_components
        self.components_ = components[:n_components]
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("PCA must be fitted before transform()")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PCA was fitted with "
                f"{self.n_features_} features"
            )

        X_centered = X - self.mean_
        X_transformed = X_centered @ self.components_.T

        return X_transformed

    def inverse_transform(self, X):
        if not self.fitted:
            raise ValueError("PCA must be fitted before inverse_transform()")

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_components_:
            raise ValueError(
                f"X has {X.shape[1]} components, but PCA was fitted with "
                f"{self.n_components_} components"
            )

        X_original = X @ self.components_ + self.mean_

        return X_original
