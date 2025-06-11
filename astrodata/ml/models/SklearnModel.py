from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import is_classifier

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseModel import BaseModel


class SklearnModel(BaseModel):
    def __init__(self, model_class, **model_params):
        self.model_class = model_class
        self.model_params = model_params
        self.model_ = None

    def get_params(self, **kwargs):
        params = {"model_class": self.model_class}
        params.update(self.model_params)
        return params

    def set_params(self, **params):
        if "model_class" in params:
            self.model_class = params.pop("model_class")
        self.model_params.update(params)
        return self

    def save(self, filepath, **kwargs):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        joblib.dump(self.model_, filepath, **kwargs)

    def load(self, filepath, **kwargs):
        self.model_ = joblib.load(filepath, **kwargs)

    def fit(self, X, y, **fit_params):
        self.model_ = self.model_class(**self.model_params)
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return pd.Series(self.model_.predict(X, **predict_params))

    def predict_proba(self, X, **predict_params):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if hasattr(self.model_, "predict_proba"):
            return pd.DataFrame(self.model_.predict_proba(X, **predict_params))
        raise AttributeError(f"{type(self.model_)} does not support predict_proba.")

    def score(self, X, y, **kwargs):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.score(X, y, **kwargs)

    def get_metrics(self, X, y, metrics: List[BaseMetric] = None):
        y_pred = self.predict(X)
        y_pred_proba = (
            self.predict_proba(X) if hasattr(self.model_, "predict_proba") else None
        )
        results = {}
        for metric in metrics:
            try:
                score = metric(y, y_pred_proba)
            except ValueError:
                score = metric(y, y_pred)
            results[metric.get_name()] = score
        return results

    def get_loss_history(self):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if hasattr(self.model_, "train_score_"):
            return self.model_.train_score_
        raise AttributeError(f"{type(self.model_)} does not have 'train_score_'.")

    def get_loss_history_metric(self, X=None, y=None, metric: BaseMetric = None):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if metric and self.has_loss_history:
            if X is None or y is None:
                raise ValueError("X and y required for metric history.")
            try:
                return [
                    metric(y, y_pred) for y_pred in self.model_.staged_predict_proba(X)
                ]
            except ValueError:
                return [metric(y, y_pred) for y_pred in self.model_.staged_predict(X)]
        else:
            raise AttributeError(f"{type(self.model_)} does not support loss history.")

    @property
    def has_loss_history(self):
        return self.model_ is not None and hasattr(self.model_, "train_score_")

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.model_params.items())
        return f"{self.__class__.__name__}(model_class={self.model_class.__name__}, {params})"
