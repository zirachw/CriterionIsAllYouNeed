import numpy as np
from abc import ABC, abstractmethod
from ..base import BaseClassifier

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, n_samples=0, impurity=0.0, counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.impurity = impurity
        self.counts = counts

class DecisionTree(BaseClassifier, ABC):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.criterion = criterion
        self.root = None
        self.isfitted = False
        self.n_features_in_ = None
        self._rng = None
        self.feature_names_in_ = None

    def fit(self, X, y, sample_weight=None, feature_names=None):
        if self.random_state is not None:
            self._rng = np.random.RandomState(self.random_state)
        else:
            self._rng = np.random.RandomState()

        self.n_features = X.shape[1]
        self.n_features_in_ = X.shape[1]
        
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        elif hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"Feature {i}" for i in range(X.shape[1])]

        X = np.asarray(X).astype(float)
        y = np.asarray(y).ravel()

        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray(sample_weight).ravel()

        self.root = self._grow_tree(X, y, sample_weight)
        self.isfitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        X = np.asarray(X).astype(float)
        if np.ndim(X) != 2:
            raise ValueError("X must be a 2D array")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, sample_weight, depth=0):
        n_samples, n_features = X.shape

        current_impurity = 0.0
        if hasattr(self, '_gini') and self.criterion == 'gini':
             current_impurity = self._gini(y, sample_weight)
        elif hasattr(self, '_entropy') and self.criterion == 'entropy':
             current_impurity = self._entropy(y, sample_weight)

        unique_classes = np.unique(y)
        node_counts = {}
        for cls in unique_classes:
            node_counts[cls] = np.sum(sample_weight[y == cls])

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split:
            return Node(value=self._calculate_leaf_value(y, sample_weight), 
                        n_samples=n_samples, impurity=current_impurity, counts=node_counts)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, sample_weight)

        if best_feature is None:
            return Node(value=self._calculate_leaf_value(y, sample_weight), 
                        n_samples=n_samples, impurity=current_impurity, counts=node_counts)

        # Create child nodes
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs

        # Check min_samples_leaf constraint
        if np.sum(left_idxs) < self.min_samples_leaf or np.sum(right_idxs) < self.min_samples_leaf:
            return Node(value=self._calculate_leaf_value(y, sample_weight), 
                        n_samples=n_samples, impurity=current_impurity, counts=node_counts)

        left = self._grow_tree(X[left_idxs], y[left_idxs], sample_weight[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], sample_weight[right_idxs], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right, 
                    value=None, n_samples=n_samples, impurity=current_impurity, counts=node_counts)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            if isinstance(node.value, dict):
                return node.value['class']
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _traverse_tree_proba(self, x, node):
        if node.value is not None:
            if isinstance(node.value, dict):
                return node.value['proba']
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree_proba(x, node.left)
        return self._traverse_tree_proba(x, node.right)

    def _best_split(self, X, y, sample_weight):
        best_gain = float('-inf')
        best_feature = None
        best_threshold = None

        for feature in range(self.n_features):
            sorted_unique = np.sort(np.unique(X[:, feature]))

            if len(sorted_unique) <= 1:
                continue

            # Use midpoints between consecutive unique values
            if len(sorted_unique) > 20:
                # Sample thresholds for efficiency
                quantiles = np.linspace(0, 1, 21)
                sorted_unique = np.quantile(X[:, feature], quantiles)
                sorted_unique = np.unique(sorted_unique)

            thresholds = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0

            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold, sample_weight)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                elif gain == best_gain and self._rng is not None:
                    # Random tie-breaking
                    if self._rng.rand() < 0.5:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold

    @abstractmethod
    def _calculate_leaf_value(self, y, sample_weight):
        """Calculate the predicted value for a leaf node."""
        pass

    @abstractmethod
    def _information_gain(self, y, X_column, threshold, sample_weight):
        """Calculate the information gain from a split."""
        pass
    
    def _check_is_fitted(self):
        if not self.isfitted:
            raise ValueError("You must call `fit` before `predict`")