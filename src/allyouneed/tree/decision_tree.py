import numpy as np
from abc import ABC, abstractmethod

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree(ABC):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.isfitted = False

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)
        self.isfitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        if np.ndim(X) != 2:
            raise ValueError("X must be a 2D array")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            return Node(value=self._calculate_leaf_value(y))

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return Node(value=self._calculate_leaf_value(y))

        # Create child nodes
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        # Check min_samples_leaf constraint
        if np.sum(left_idxs) < self.min_samples_leaf or np.sum(right_idxs) < self.min_samples_leaf:
            return Node(value=self._calculate_leaf_value(y))

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _best_split(self, X, y):
        best_gain = float('-inf')
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    @abstractmethod
    def _calculate_leaf_value(self, y):
        """Calculate the predicted value for a leaf node."""
        pass
    
    @abstractmethod
    def _information_gain(self, y, X_column, threshold):
        """Calculate the information gain from a split."""
        pass
    
    def _check_is_fitted(self):
        if not self.isfitted:
            raise ValueError("You must call `fit` before `predict`")