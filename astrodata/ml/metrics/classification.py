from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from astrodata.ml.metrics.BaseMetric import BaseMetric


class Accuracy(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        return accuracy_score(y_true, y_pred, normalize=kwargs.get("normalize", True))

    def get_name(self):
        return "accuracy"


class F1Score(BaseMetric):
    def __init__(self, average="macro"):
        self.average = average

    def __call__(self, y_true, y_pred, **kwargs):
        return f1_score(y_true, y_pred, average=self.average)

    def get_name(self):
        return "f1"


class ConfusionMatrix(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        return confusion_matrix(y_true, y_pred, labels=kwargs.get("labels", None))

    def get_name(self):
        return "confusion_matrix"


class Precision(BaseMetric):
    def __init__(self, average="macro"):
        self.average = average

    def __call__(self, y_true, y_pred, **kwargs):
        return precision_score(y_true, y_pred, average=self.average)

    def get_name(self):
        return "precision"


class Recall(BaseMetric):
    def __init__(self, average="macro"):
        self.average = average

    def __call__(self, y_true, y_pred, **kwargs):
        return recall_score(y_true, y_pred, average=self.average)

    def get_name(self):
        return "recall"
