import numpy as np
from .decision_tree import DecisionTree, Node

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None,
                 class_weight=None, criterion='gini'):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, random_state, criterion)
        self.class_weight = class_weight
        self.n_classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Compute sample weights from class weights
        if self.class_weight is not None:
            sample_weight = self._compute_sample_weight(y)
        else:
            sample_weight = None

        super().fit(X, y, sample_weight=sample_weight)
        return self

    def _compute_sample_weight(self, y):
        if self.class_weight == 'balanced':
            # Balanced: inversely proportional to class frequencies
            class_counts = np.bincount(y)
            n_samples = len(y)
            n_classes = len(self.classes_)
            weights = n_samples / (n_classes * class_counts)
            sample_weight = weights[y]
        elif isinstance(self.class_weight, dict):
            # Custom weights per class
            sample_weight = np.array([self.class_weight.get(cls, 1.0) for cls in y])
        else:
            sample_weight = np.ones(len(y))

        return sample_weight

    def _calculate_leaf_value(self, y, sample_weight):
        """Returns dict with class probabilities and majority class"""
        weighted_votes = {}
        for cls, weight in zip(y, sample_weight):
            weighted_votes[cls] = weighted_votes.get(cls, 0) + weight

        total_weight = sum(weighted_votes.values())
        proba = np.zeros(self.n_classes_)
        for i, cls in enumerate(self.classes_):
            if cls in weighted_votes:
                proba[i] = weighted_votes[cls] / total_weight

        majority_class = max(weighted_votes, key=weighted_votes.get)
        return {'class': majority_class, 'proba': proba}

    def predict_proba(self, X):
        """Predict class probabilities"""
        self._check_is_fitted()
        X = np.asarray(X).astype(float)
        if np.ndim(X) != 2:
            raise ValueError("X must be a 2D array")
        return np.array([self._traverse_tree_proba(x, self.root) for x in X])

    def _information_gain(self, y, X_column, threshold, sample_weight):
        if self.criterion == 'gini':
            parent_impurity = self._gini(y, sample_weight)
        else:
            parent_impurity = self._entropy(y, sample_weight)

        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0

        total_weight = np.sum(sample_weight)
        weight_left = np.sum(sample_weight[left_idxs])
        weight_right = np.sum(sample_weight[right_idxs])

        if self.criterion == 'gini':
            impurity_left = self._gini(y[left_idxs], sample_weight[left_idxs])
            impurity_right = self._gini(y[right_idxs], sample_weight[right_idxs])
        else:
            impurity_left = self._entropy(y[left_idxs], sample_weight[left_idxs])
            impurity_right = self._entropy(y[right_idxs], sample_weight[right_idxs])

        child_impurity = (weight_left / total_weight) * impurity_left + \
                        (weight_right / total_weight) * impurity_right

        return parent_impurity - child_impurity

    def _gini(self, y, sample_weight):
        """Weighted Gini impurity calculation"""
        unique_classes = np.unique(y)
        total_weight = np.sum(sample_weight)

        gini = 1.0
        for cls in unique_classes:
            cls_mask = (y == cls)
            cls_weight = np.sum(sample_weight[cls_mask])
            if cls_weight > 0:
                p = cls_weight / total_weight
                gini -= p * p

        return gini

    def _entropy(self, y, sample_weight):
        """Weighted entropy calculation"""
        unique_classes = np.unique(y)
        total_weight = np.sum(sample_weight)

        entropy = 0.0
        for cls in unique_classes:
            cls_mask = (y == cls)
            cls_weight = np.sum(sample_weight[cls_mask])
            if cls_weight > 0:
                p = cls_weight / total_weight
                entropy -= p * np.log2(p)

        return entropy