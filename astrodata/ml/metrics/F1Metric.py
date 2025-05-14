from sklearn.metrics import f1_score
from astrodata.ml.metrics.BaseMetric import BaseMetric

class F1Metric(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        return f1_score(y_true, y_pred)
    def get_name(self):
        return "f1"