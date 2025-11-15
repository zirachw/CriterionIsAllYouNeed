import numpy as np
from .base import BaseEncoder


class LabelEncoder(BaseEncoder):

    def __init__(self):
        self.class_to_int = {}
        self.int_to_class = {}
        self.fitted = False

    def fit(self, y):
        classes = np.unique(y)
        self.class_to_int = {c: i for i, c in enumerate(classes)}
        self.int_to_class = {i: c for c, i in self.class_to_int.items()}
        self.fitted = True
        return self

    def transform(self, y):
        if not self.fitted:
            raise ValueError("LabelEncoder must be fitted before transform()")

        result = []
        for c in y:
            if c not in self.class_to_int:
                raise ValueError(f"Unknown label: {c}. Known labels: {list(self.class_to_int.keys())}")
            result.append(self.class_to_int[c])
        return np.array(result, dtype=int)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_int):
        if not self.fitted:
            raise ValueError("LabelEncoder must be fitted before inverse_transform()")
        return np.array([self.int_to_class[i] for i in y_int])
