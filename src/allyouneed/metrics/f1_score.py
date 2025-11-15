from .base import Metric
import numpy as np

class F1Score(Metric):
    def __init__(self, average="macro"):
        assert average in ["macro", "micro", "weighted"], "Invalid average"
        self.average = average

    def __call__(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        labels = np.unique(np.concatenate([y_true, y_pred]))

        tp = np.array([np.sum((y_true == c) & (y_pred == c)) for c in labels])
        fp = np.array([np.sum((y_true != c) & (y_pred == c)) for c in labels])
        fn = np.array([np.sum((y_true == c) & (y_pred != c)) for c in labels])

        # Precision, Recall
        precision = np.where(tp + fp == 0, 0, tp / (tp + fp))
        recall    = np.where(tp + fn == 0, 0, tp / (tp + fn))

        # F1
        f1 = np.where(
            precision + recall == 0,
            0,
            2 * precision * recall / (precision + recall)
        )

        # === Handle average ===
        if self.average == "macro":
            return float(np.mean(f1))

        elif self.average == "micro":
            tp_sum = tp.sum()
            fp_sum = fp.sum()
            fn_sum = fn.sum()
            prec = tp_sum / (tp_sum + fp_sum)
            rec  = tp_sum / (tp_sum + fn_sum)
            return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

        elif self.average == "weighted":
            weights = np.array([np.sum(y_true == c) for c in labels])
            return float(np.average(f1, weights=weights))
