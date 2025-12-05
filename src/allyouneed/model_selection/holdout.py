import numpy as np
from .base import BaseCrossValidator


class Holdout(BaseCrossValidator):
    """Hold-out cross-validator."""

    def __init__(self, test_size=0.2, shuffle=True, random_state=None):
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n_samples = len(X)
        n_test = int(n_samples * self.test_size)
        n_train = n_samples - n_test

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        yield train_indices, test_indices

    def train_test_split(self, X, y=None):
        X = np.asarray(X)
        train_indices, test_indices = next(self.split(X, y))
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        
        if y is not None:
            y = np.asarray(y)
            y_train = y[train_indices]
            y_test = y[test_indices]
            return X_train, X_test, y_train, y_test
        
        return X_train, X_test


class StratifiedHoldout(BaseCrossValidator):
    """Stratified Hold-out cross-validator."""

    def __init__(self, test_size=0.2, shuffle=True, random_state=None):
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        if y is None:
            raise ValueError("y is required for StratifiedHoldout")

        y = np.asarray(y)
        n_samples = len(X)

        unique_classes, y_inv = np.unique(y, return_inverse=True)
        n_classes = len(unique_classes)
        class_counts = np.bincount(y_inv)

        train_indices = []
        test_indices = []

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)

        for class_idx in range(n_classes):
            class_mask = (y_inv == class_idx)
            class_indices = np.where(class_mask)[0]
            n_class_samples = len(class_indices)

            if self.shuffle:
                rng.shuffle(class_indices)

            n_test = int(n_class_samples * self.test_size)
            n_train = n_class_samples - n_test

            train_indices.extend(class_indices[:n_train])
            test_indices.extend(class_indices[n_train:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if self.shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)

        yield train_indices, test_indices

    def train_test_split(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        train_indices, test_indices = next(self.split(X, y))
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
