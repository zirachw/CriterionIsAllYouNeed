from abc import ABC, abstractmethod


class BaseEncoder(ABC):

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X):
        return self.fit(X).transform(X)
