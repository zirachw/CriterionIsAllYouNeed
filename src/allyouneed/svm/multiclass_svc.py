import numpy as np
from .svc import SVC
from ..base import BaseClassifier


class MulticlassSVC(BaseClassifier):
    def __init__(self, kernel='linear', gamma=None, degree=3, coef0=0.0, C=1.0,
                 class_weight=None, optimizer='cvxopt', max_iter=200, tol=0.001, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.C = C
        self.class_weight = class_weight
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.classes_ = None
        self.classifiers_ = []
        self.class_pairs_ = []
        self.n_features_in_ = None
        self.isfitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need at least 2 classes")
        if n_classes == 2:
            raise ValueError("Use SVC for binary classification")

        self.classifiers_ = []
        self.class_pairs_ = []

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                class_i = self.classes_[i]
                class_j = self.classes_[j]

                mask = (y == class_i) | (y == class_j)
                X_pair = X[mask]
                y_pair = y[mask]

                svc = SVC(
                    kernel=self.kernel,
                    gamma=self.gamma,
                    degree=self.degree,
                    coef0=self.coef0,
                    C=self.C,
                    class_weight=self.class_weight,
                    optimizer=self.optimizer,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state
                )
                svc.fit(X_pair, y_pair)

                self.classifiers_.append(svc)
                self.class_pairs_.append((class_i, class_j))

        self.isfitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        n_samples = len(X)
        n_classes = len(self.classes_)

        votes = np.zeros((n_samples, n_classes), dtype=int)
        confidence = np.zeros((n_samples, n_classes), dtype=float)

        for svc, (class_i, class_j) in zip(self.classifiers_, self.class_pairs_):
            predictions = svc.predict(X)
            decision_values = svc._decision_function(X)

            class_i_idx = np.where(self.classes_ == class_i)[0][0]
            class_j_idx = np.where(self.classes_ == class_j)[0][0]

            for k in range(n_samples):
                if predictions[k] == class_i:
                    votes[k, class_i_idx] += 1
                    confidence[k, class_i_idx] += abs(decision_values[k])
                else:
                    votes[k, class_j_idx] += 1
                    confidence[k, class_j_idx] += abs(decision_values[k])

        results = np.zeros(n_samples, dtype=self.classes_.dtype)
        for i in range(n_samples):
            max_votes = np.max(votes[i])
            tied_classes = np.where(votes[i] == max_votes)[0]

            if len(tied_classes) == 1:
                results[i] = self.classes_[tied_classes[0]]
            else:
                winner = tied_classes[np.argmax(confidence[i, tied_classes])]
                results[i] = self.classes_[winner]

        return results

    def _check_is_fitted(self):
        if not self.isfitted:
            raise ValueError("You must call `fit` before `predict`")
