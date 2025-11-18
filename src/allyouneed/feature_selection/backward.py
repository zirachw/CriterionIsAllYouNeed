import numpy as np
from typing import Callable, Union
from concurrent.futures import ThreadPoolExecutor
from .base import BaseFeatureSelector
from ..metrics.base import Metric
from ..model_selection.base import BaseCrossValidator


class BackwardFeatureElimination(BaseFeatureSelector):

    def __init__(
        self,
        estimator,
        n_features_to_select: int,
        scoring: Union[Metric, Callable],
        cv: BaseCrossValidator,
        n_jobs: int = 1,
        verbose: bool = True,
        feature_names: list = None
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.feature_names = feature_names
        self.selected_features_ = None

    def _get_feature_name(self, idx):
        if self.feature_names is not None:
            return self.feature_names[idx]
        return f"Feature_{idx}"

    def _evaluate_features(self, X, y):
        scores = []
        for train_idx, val_idx in self.cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            self.estimator.fit(X_train, y_train)
            y_pred = self.estimator.predict(X_val)
            score = self.scoring(y_val, y_pred)
            scores.append(score)

        return np.mean(scores)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        if self.verbose:
            print("=" * 60)
            print("Backward Feature Elimination")
            print(f"Total features: {n_features}")
            print(f"Target features: {self.n_features_to_select}")
            print("=" * 60)

        selected = list(range(n_features))

        while len(selected) > self.n_features_to_select:
            iteration = n_features - len(selected) + 1
            best_score = -np.inf
            least_important_feature = None

            if self.verbose:
                print(f"\nIteration {iteration}/{n_features - self.n_features_to_select}")
                print(f"Testing removal of {len(selected)} features...")

            if self.n_jobs == 1:
                # Sequential execution
                for feature in selected:
                    current_features = [f for f in selected if f != feature]
                    X_subset = X[:, current_features]
                    score = self._evaluate_features(X_subset, y)

                    if score > best_score:
                        best_score = score
                        least_important_feature = feature
            else:
                # Parallel execution
                def evaluate_removal(feature):
                    current_features = [f for f in selected if f != feature]
                    X_subset = X[:, current_features]
                    score = self._evaluate_features(X_subset, y)
                    return feature, score

                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    results = list(executor.map(evaluate_removal, selected))

                for feature, score in results:
                    if score > best_score:
                        best_score = score
                        least_important_feature = feature

            selected.remove(least_important_feature)

            if self.verbose:
                feature_name = self._get_feature_name(least_important_feature)
                print(f"Removed: {feature_name} (score without it: {best_score:.4f})")
                remaining_names = [self._get_feature_name(f) for f in selected]
                print(f"Remaining features ({len(selected)}): {remaining_names}")

        self.selected_features_ = np.array(selected)

        if self.verbose:
            print("\n" + "=" * 60)
            print("Feature elimination completed!")
            selected_names = [self._get_feature_name(f) for f in self.selected_features_]
            print(f"Final selected features: {selected_names}")
            print("=" * 60)

        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("You must call fit before transform")

        # Handle pandas DataFrame
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, self.selected_features_].values
        except ImportError:
            pass

        # Handle numpy array or convert to numpy
        X = np.asarray(X)
        return X[:, self.selected_features_]
