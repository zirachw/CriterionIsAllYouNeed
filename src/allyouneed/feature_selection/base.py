from abc import ABC, abstractmethod
from ..base import BaseEstimator, TransformerMixin


class BaseFeatureSelector(BaseEstimator, TransformerMixin, ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass
