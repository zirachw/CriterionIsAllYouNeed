from abc import ABC, abstractmethod

class BaseCrossValidator(ABC):
    @abstractmethod
    def split(self, X, y=None):
        raise NotImplementedError("Subclasses should implement this method.")
    
