import numpy as np
from .base import BaseFeatureSelector


class ForwardFeatureSelection(BaseFeatureSelector):

    def __init__(self, estimator, n_features_to_select):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.selected_features_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        selected = []
        remaining = list(range(n_features))

        while len(selected) < self.n_features_to_select:
            best_score = -np.inf
            best_feature = None

            for feature in remaining:
                current_features = selected + [feature]
                X_subset = X[:, current_features]

                self.estimator.fit(X_subset, y)
                y_pred = self.estimator.predict(X_subset)
                score = np.mean(y_pred == y)

                if score > best_score:
                    best_score = score
                    best_feature = feature

            selected.append(best_feature)
            remaining.remove(best_feature)

        self.selected_features_ = np.array(selected)
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("You must call fit before transform")
        return np.asarray(X)[:, self.selected_features_]
