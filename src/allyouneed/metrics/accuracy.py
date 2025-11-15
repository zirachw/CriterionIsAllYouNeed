from .base import Metric
import numpy as np

class Accuracy(Metric):
    def __call__(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(y_true == y_pred))