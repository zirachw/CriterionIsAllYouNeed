from abc import ABC, abstractmethod
from ..base import BaseEstimator, TransformerMixin


class BaseEncoder(BaseEstimator, TransformerMixin, ABC):

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass
