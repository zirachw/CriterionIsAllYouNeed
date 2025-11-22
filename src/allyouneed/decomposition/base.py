from abc import ABC, abstractmethod
from ..base import BaseEstimator, TransformerMixin


class BaseDecomposition(BaseEstimator, TransformerMixin, ABC):

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass
