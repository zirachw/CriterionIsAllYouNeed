import numpy as np
import matplotlib.pyplot as plt
from .decision_tree import DecisionTree, Node

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None,
                 class_weight=None, criterion='gini'):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, random_state, criterion)
        self.class_weight = class_weight
        self.n_classes_ = None

    def fit(self, X, y, feature_names=None):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Compute sample weights from class weights
        if self.class_weight is not None:
            sample_weight = self._compute_sample_weight(y)
        else:
            sample_weight = None

        super().fit(X, y, sample_weight=sample_weight, feature_names=feature_names)
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

    def visualize_tree(self, filename="tree_visualization.png", top_n=None):
        if self.root is None:
            return

        max_tree_depth = self._get_depth(self.root)

        if top_n is None:
            visual_depth = max_tree_depth
        elif top_n < 0:
            raise ValueError(f"Parameter top_n tidak boleh negatif. Nilai yang diterima: {top_n}")
        elif top_n == 0:
            visual_depth = 0
        elif top_n > max_tree_depth:
            visual_depth = max_tree_depth
        else:
            visual_depth = top_n

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_axis_off()
        
        self._plot_node(ax, self.root, x=0.5, y=1.0, dx=0.5, dy=1.0/(visual_depth+1), 
                        depth=0, max_depth=visual_depth)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def _plot_node(self, ax, node, x, y, dx, dy, depth, max_depth):
        if node is None:
            return

        val_str = "N/A"
        if node.counts:
            counts_list = [node.counts.get(c, 0) for c in self.classes_]
            val_str = str([int(v) if isinstance(v, (int, np.integer)) else float(f"{v:.1f}") for v in counts_list])

        content = f"{self.criterion} = {node.impurity:.3f}\n"
        content += f"samples = {node.n_samples}\n"
        content += f"value = {val_str}\n"

        if node.value is not None and (node.left is None and node.right is None):
            if isinstance(node.value, dict):
                 content += f"class = {node.value['class']}"
            else:
                 content += f"class = {node.value}"
            
            bbox_props = dict(boxstyle="round,pad=0.5", fc="#e5f5e0", ec="black", alpha=0.9)
            text = content
        else:
            if isinstance(node.value, dict):
                 content += f"class = {node.value['class']}"
            elif node.value is not None:
                 content += f"class = {node.value}"
            
            feature_name = self.feature_names_in_[node.feature] if self.feature_names_in_ else f"Feat {node.feature}"
            header = f"{feature_name} < {node.threshold:.2f}\n"
            text = header + content
            bbox_props = dict(boxstyle="round,pad=0.5", fc="#e0f7fa", ec="black", alpha=0.9)

        ax.text(x, y, text, ha="center", va="center", bbox=bbox_props, fontsize=8, family='monospace')

        if (node.left is None and node.right is None) or depth >= max_depth:
            return

        y_next = y - dy
        x_left = x - (dx / 2)
        x_right = x + (dx / 2)

        ax.plot([x, x_left], [y, y_next], 'k-', lw=1, zorder=-1)
        ax.plot([x, x_right], [y, y_next], 'k-', lw=1, zorder=-1)

        mid_y = (y + y_next) / 2
        mid_x_left = (x + x_left) / 2
        mid_x_right = (x + x_right) / 2
        
        ax.text(mid_x_left, mid_y, "True", ha="right", va="center", fontsize=7, color="blue", weight='bold')
        ax.text(mid_x_right, mid_y, "False", ha="left", va="center", fontsize=7, color="red", weight='bold')

        self._plot_node(ax, node.left, x_left, y_next, dx/2, dy, depth + 1, max_depth)
        self._plot_node(ax, node.right, x_right, y_next, dx/2, dy, depth + 1, max_depth)

    def _get_depth(self, node):
        if node is None or (node.left is None and node.right is None):
            return 0
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))