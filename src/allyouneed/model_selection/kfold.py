import numpy as np
from .base import BaseCrossValidator


class KFold(BaseCrossValidator):
    """K-Fold cross-validator."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError(
                f"n_splits={self.n_splits} cannot be greater than "
                f"number of samples={n_samples}"
            )

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

    def get_n_splits(self):
        return self.n_splits


class StratifiedKFold(BaseCrossValidator):
    """Stratified K-Fold cross-validator."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        if y is None:
            raise ValueError("y is required for StratifiedKFold")

        y = np.asarray(y)
        n_samples = len(X)

        if self.n_splits > n_samples:
            raise ValueError(
                f"n_splits={self.n_splits} cannot be greater than "
                f"number of samples={n_samples}"
            )

        unique_classes, y_inv = np.unique(y, return_inverse=True)
        n_classes = len(unique_classes)
        class_counts = np.bincount(y_inv)

        if np.any(self.n_splits > class_counts):
            raise ValueError(
                f"n_splits={self.n_splits} cannot be greater than "
                f"the number of members in each class"
            )

        test_folds = np.zeros(n_samples, dtype=int)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)

        for class_idx in range(n_classes):
            class_mask = (y_inv == class_idx)
            class_indices = np.where(class_mask)[0]
            n_class_samples = len(class_indices)

            if self.shuffle:
                rng.shuffle(class_indices)

            fold_assignment = np.arange(n_class_samples) % self.n_splits
            test_folds[class_indices] = fold_assignment

        for fold_idx in range(self.n_splits):
            test_mask = (test_folds == fold_idx)
            test_indices = np.where(test_mask)[0]
            train_indices = np.where(~test_mask)[0]
            yield train_indices, test_indices

    def get_n_splits(self):
        return self.n_splits
