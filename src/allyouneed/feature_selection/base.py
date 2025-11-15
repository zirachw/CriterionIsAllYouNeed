from abc import ABC, abstractmethod


class BaseFeatureSelector(ABC):

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
