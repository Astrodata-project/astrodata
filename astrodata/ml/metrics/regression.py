import numpy as np

from astrodata.ml.metrics.BaseMetric import BaseMetric


class MAE(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    def get_name(self):
        return "mae"


class MSE(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.mean((y_true - y_pred) ** 2)

    def get_name(self):
        return "mse"


class R2(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        # To avoid division by zero, handle the degenerate case:
        if ss_tot == 0:
            return 0.0  # or float('nan'), depending on your convention
        return 1 - ss_res / ss_tot

    def get_name(self):
        return "r2"


class RMSE(BaseMetric):
    def __call__(self, y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def get_name(self):
        return "rmse"
