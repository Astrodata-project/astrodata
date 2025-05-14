from astrodata.ml.metrics.BaseMetric import BaseMetric
import numpy as np

class AccuracyMetric(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return (y_true == y_pred).mean()
    def get_name(self):
        return "accuracy"
