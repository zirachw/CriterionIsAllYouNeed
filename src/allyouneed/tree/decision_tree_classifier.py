import numpy as np
from .decision_tree import DecisionTree

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None, class_weight=None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, random_state)
        self.class_weight = class_weight

    def fit(self, X, y):
        self.classes_ = np.unique(y)

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
        """Returns the class with highest weighted vote"""
        weighted_votes = {}
        for cls, weight in zip(y, sample_weight):
            weighted_votes[cls] = weighted_votes.get(cls, 0) + weight
        return max(weighted_votes, key=weighted_votes.get)

    def _information_gain(self, y, X_column, threshold, sample_weight):
        parent_entropy = self._entropy(y, sample_weight)

        left_idxs = X_column < threshold
        right_idxs = ~left_idxs

        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0

        # Weighted entropy for children
        total_weight = np.sum(sample_weight)
        weight_left = np.sum(sample_weight[left_idxs])
        weight_right = np.sum(sample_weight[right_idxs])

        e_l = self._entropy(y[left_idxs], sample_weight[left_idxs])
        e_r = self._entropy(y[right_idxs], sample_weight[right_idxs])

        child_entropy = (weight_left / total_weight) * e_l + (weight_right / total_weight) * e_r

        return parent_entropy - child_entropy

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