from abc import ABC, abstractmethod

class Metric(ABC):

    @abstractmethod
    def __call__(self, y_true, y_pred):
        pass