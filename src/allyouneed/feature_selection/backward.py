import numpy as np
from .base import BaseFeatureSelector


class BackwardFeatureElimination(BaseFeatureSelector):

    def __init__(self, estimator, n_features_to_select):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.selected_features_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]

        selected = list(range(n_features))

        while len(selected) > self.n_features_to_select:
            worst_score = np.inf
            worst_feature = None

            for feature in selected:
                current_features = [f for f in selected if f != feature]
                X_subset = X[:, current_features]

                self.estimator.fit(X_subset, y)
                y_pred = self.estimator.predict(X_subset)
                score = np.mean(y_pred == y)

                if score < worst_score:
                    worst_score = score
                    worst_feature = feature

            selected.remove(worst_feature)

        self.selected_features_ = np.array(selected)
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("You must call fit before transform")
        return np.asarray(X)[:, self.selected_features_]
