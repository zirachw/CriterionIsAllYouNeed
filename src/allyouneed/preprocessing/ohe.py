import numpy as np
import pandas as pd
from .base import BaseEncoder


class OneHotEncoder(BaseEncoder):

    def __init__(self, columns=None, handle_unknown="ignore", dtype=float):
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.dtype = dtype
        self.categories_ = {}
        self.feature_names_ = []
        self.fitted = False

    def fit(self, X):
        X = self._to_dataframe(X)

        if self.columns is None:
            self.columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in self.columns:
            self.categories_[col] = np.unique(X[col].astype(str)).tolist()

        self.feature_names_ = []
        for col in self.columns:
            for cat in self.categories_[col]:
                self.feature_names_.append(f"{col}__{cat}")

        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise ValueError("OneHotEncoder must be fitted before transform()")

        X = self._to_dataframe(X)
        encoded_arrays = []

        for col in self.columns:
            values = X[col].astype(str).values
            categories = self.categories_[col]
            mat = np.zeros((len(values), len(categories)), dtype=self.dtype)

            for i, val in enumerate(values):
                if val in categories:
                    j = categories.index(val)
                    mat[i, j] = 1.0
                else:
                    if self.handle_unknown == "error":
                        raise ValueError(f"Unknown category '{val}' in column '{col}'")

            encoded_arrays.append(mat)

        if encoded_arrays:
            return np.hstack(encoded_arrays).astype(self.dtype)
        else:
            return np.zeros((len(X), 0), dtype=self.dtype)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, encoded):
        if not self.fitted:
            raise ValueError("OneHotEncoder must be fitted before inverse_transform()")

        encoded = np.asarray(encoded)
        idx = 0
        outputs = {}

        for col in self.columns:
            num_cats = len(self.categories_[col])
            slice_col = encoded[:, idx: idx + num_cats]
            idx += num_cats

            inv = []
            for row in slice_col:
                if np.sum(row) == 0:
                    inv.append(None)
                else:
                    j = np.argmax(row)
                    inv.append(self.categories_[col][j])

            outputs[col] = inv

        return pd.DataFrame(outputs)

    def _to_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            return pd.DataFrame(X)
        return X
