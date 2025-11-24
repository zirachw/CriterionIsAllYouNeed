import numpy as np
from typing import Callable, Union
from concurrent.futures import ThreadPoolExecutor
from .base import BaseFeatureSelector
from ..metrics.base import Metric
from ..model_selection.base import BaseCrossValidator


class BackwardForwardFeatureSelection(BaseFeatureSelector):

    def __init__(
        self,
        estimator,
        n_features_to_select: int,
        scoring: Union[Metric, Callable],
        cv: BaseCrossValidator,
        n_jobs: int = 1,
        verbose: bool = True,
        feature_names: list = None,
        patience: int = 3
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.feature_names = feature_names
        self.patience = patience
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
            print("Backward-Forward Feature Selection")
            print(f"Total features: {n_features}")
            print(f"Target features: {self.n_features_to_select}")
            print(f"Patience: {self.patience}")
            print("=" * 60)

        selected = list(range(n_features))
        eliminated = []
        iteration = 0
        no_improvement_count = 0

        current_score = self._evaluate_features(X[:, selected], y)
        if self.verbose:
            print(f"\nInitial score with all features: {current_score:.4f}")

        while len(selected) > self.n_features_to_select:
            iteration += 1
            improved = False

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Iteration {iteration}")
                print(f"Current features: {len(selected)}, Eliminated: {len(eliminated)}")
                print(f"Current score: {current_score:.4f}")

            # BACKWARD STEP
            if self.verbose:
                print(f"\n[BACKWARD] Testing removal of {len(selected)} features...")

            best_score_backward = -np.inf
            best_feature_to_remove = None

            if self.n_jobs == 1:
                for feature in selected:
                    temp_selected = [f for f in selected if f != feature]
                    X_subset = X[:, temp_selected]
                    score = self._evaluate_features(X_subset, y)

                    if score > best_score_backward:
                        best_score_backward = score
                        best_feature_to_remove = feature
            else:
                def evaluate_removal(feature):
                    temp_selected = [f for f in selected if f != feature]
                    X_subset = X[:, temp_selected]
                    score = self._evaluate_features(X_subset, y)
                    return feature, score

                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    results = list(executor.map(evaluate_removal, selected))

                for feature, score in results:
                    if score > best_score_backward:
                        best_score_backward = score
                        best_feature_to_remove = feature

            if best_feature_to_remove is not None:
                selected.remove(best_feature_to_remove)
                eliminated.append(best_feature_to_remove)

                if self.verbose:
                    fname = self._get_feature_name(best_feature_to_remove)
                    print(f"Removed: {fname} (score: {best_score_backward:.4f})")

                if best_score_backward > current_score:
                    current_score = best_score_backward
                    improved = True
                    no_improvement_count = 0
                    if self.verbose:
                        print(f"Score improved: {current_score:.4f}")
                else:
                    current_score = best_score_backward

            # FORWARD STEP
            if len(eliminated) > 0 and len(selected) > self.n_features_to_select:
                if self.verbose:
                    print(f"\n[FORWARD] Testing addition of {len(eliminated)} eliminated features...")

                best_score_forward = current_score
                best_feature_to_add = None

                if self.n_jobs == 1:
                    for feature in eliminated:
                        temp_selected = selected + [feature]
                        X_subset = X[:, temp_selected]
                        score = self._evaluate_features(X_subset, y)

                        if score > best_score_forward:
                            best_score_forward = score
                            best_feature_to_add = feature
                else:
                    def evaluate_addition(feature):
                        temp_selected = selected + [feature]
                        X_subset = X[:, temp_selected]
                        score = self._evaluate_features(X_subset, y)
                        return feature, score

                    with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                        results = list(executor.map(evaluate_addition, eliminated))

                    for feature, score in results:
                        if score > best_score_forward:
                            best_score_forward = score
                            best_feature_to_add = feature

                if best_feature_to_add is not None and best_score_forward > current_score:
                    eliminated.remove(best_feature_to_add)
                    selected.append(best_feature_to_add)
                    current_score = best_score_forward
                    improved = True
                    no_improvement_count = 0

                    if self.verbose:
                        fname = self._get_feature_name(best_feature_to_add)
                        print(f"Added back: {fname} (score: {current_score:.4f})")
                        print(f"Score improved: {current_score:.4f}")
                else:
                    if self.verbose:
                        print("No beneficial feature to add back")

            if not improved:
                no_improvement_count += 1
                if self.verbose:
                    print(f"No improvement (patience: {no_improvement_count}/{self.patience})")

                if no_improvement_count >= self.patience:
                    if self.verbose:
                        print(f"\nEarly stopping: no improvement for {self.patience} iterations")
                    break

        self.selected_features_ = np.array(sorted(selected))

        if self.verbose:
            print("\n" + "=" * 60)
            print("Feature selection completed!")
            print(f"Final features: {len(self.selected_features_)}")
            print(f"Final score: {current_score:.4f}")
            selected_names = [self._get_feature_name(f) for f in self.selected_features_]
            print(f"Selected features: {selected_names}")
            print("=" * 60)

        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("You must call fit before transform")

        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, self.selected_features_].values
        except ImportError:
            pass

        X = np.asarray(X)
        return X[:, self.selected_features_]
