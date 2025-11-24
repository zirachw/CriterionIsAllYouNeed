import numpy as np
from ..base import BaseClassifier


class LogisticRegression(BaseClassifier):
    SOLVER_TABLE = {
        "sga": "_stochastic_gradient_ascent",
        "bga": "_batch_gradient_ascent",
        "mgd": "_mini_batch_gradient_descent",
    }
    
    def __init__(
        self,
        max_iter=200,
        learning_rate=0.1,
        solver="sga",
        batch_size=32,
        random_state=None,
        tol=1e-4,
        class_weight=None,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        if solver not in self.SOLVER_TABLE:
            raise ValueError(f"Unknown solver: {solver}")
        self.solver_name = solver
        self.solver_fn = getattr(self, self.SOLVER_TABLE[solver])
        self.batch_size = batch_size
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state) if random_state is not None else np.random
        self.tol = tol
        self.class_weight = class_weight

        self._classes = None
        self._W = None
        self._isfitted = False
        
    def _logistic_function(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _compute_sample_weight(self, y):
        if self.class_weight == 'balanced':
            classes, class_counts = np.unique(y, return_counts=True)
            n_samples = len(y)
            n_classes = len(classes)
            weights = n_samples / (n_classes * class_counts)
            sample_weight = weights[np.searchsorted(classes, y)]
        elif isinstance(self.class_weight, dict):
            sample_weight = np.array([self.class_weight.get(cls, 1.0) for cls in y])
        else:
            sample_weight = np.ones(len(y))
        return sample_weight

    def _check_is_fitted(self):
        if not self._isfitted:
            raise ValueError("You must call `fit` before `predict`.")
            
    def _validate_input(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y is not None:
            y = np.asarray(y)
            if len(X) != len(y):
                raise ValueError("X and y must have the same length.")
            return X, y
        return X
        
    def _add_bias(self, X):
        bias = np.ones((X.shape[0], 1))
        return np.hstack([bias, X])
        
    def _stochastic_gradient_ascent(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        for _ in range(self.max_iter):
            indices = self._rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                w_batch = sample_weight[batch_idx]

                p = self._logistic_function(X_batch @ w)
                error = (y_batch - p) * w_batch
                gradient = X_batch.T @ error / w_batch.sum()
                w += self.learning_rate * gradient

            # cek semuanya
            p_full = self._logistic_function(X @ w)
            error_full = (y - p_full) * sample_weight
            gradient_full = X.T @ error_full / sample_weight.sum()
            if np.linalg.norm(gradient_full) < self.tol:
                break

        return w
    
    def _batch_gradient_ascent(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        for _ in range(self.max_iter):
            p = self._logistic_function(X @ w)
            error = (y - p) * sample_weight
            gradient = X.T @ error / sample_weight.sum()

            w += self.learning_rate * gradient

            if np.linalg.norm(gradient) < self.tol:
                break

        return w
    
    def _mini_batch_gradient_descent(self, X, y, sample_weight=None):
        n_samples, n_features = X.shape
        w = np.zeros(n_features)

        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        for _ in range(self.max_iter):
            indices = self._rng.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                wb = sample_weight[batch_idx]

                p = self._logistic_function(Xb @ w)
                error = (p - yb) * wb
                gradient = Xb.T @ error / wb.sum()

                w -= self.learning_rate * gradient  # descent

            # cek konvergensi full batch
            p_full = self._logistic_function(X @ w)
            error_full = (p_full - y) * sample_weight
            gradient_full = X.T @ error_full / sample_weight.sum()
            if np.linalg.norm(gradient_full) < self.tol:
                break

        return w

        
    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        self._classes = np.unique(y)
        X = self._add_bias(X)

        if self.class_weight is not None:
            sample_weight = self._compute_sample_weight(y)
        else:
            sample_weight = np.ones(len(y))

        if self._classes.size > 2:
            W_all = []
            for cls in self._classes:
                y_binary = (y == cls).astype(int)
                w = self.solver_fn(X, y_binary, sample_weight)
                W_all.append(w)
            self._W = np.vstack(W_all)

        else:
            positive_class = self._classes[-1]
            y_binary = (y == positive_class).astype(int)
            self._W = self.solver_fn(X, y_binary, sample_weight)

        self._isfitted = True
        return self
        
    def predict_proba(self, X):
        self._check_is_fitted()
        X = self._add_bias(self._validate_input(X))
        
        if self._classes.size > 2:
            logits = X @ self._W.T
            return self._logistic_function(logits)
        
        logits = X @ self._W
        proba_pos = self._logistic_function(logits)
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])
    
    def predict(self, X):
        probs = self.predict_proba(X)
        y_pred = np.argmax(probs, axis=1)
        return self._classes[y_pred]