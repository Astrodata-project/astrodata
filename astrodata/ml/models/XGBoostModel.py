from typing import List

import joblib
import pandas as pd

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseModel import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, model_class, **model_params):
        if "eval_metric" not in model_params:
            model_params["eval_metric"] = (
                "logloss" if hasattr(model_class, "predict_proba") else "rmse"
            )
        self.model_class = model_class
        self.model_params = model_params
        self.model_ = None
        self._evals_result = None

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
        if "eval_set" not in fit_params:
            fit_params["eval_set"] = [(X, y)]
        self.model_ = self.model_class(**self.model_params)
        self.model_.fit(X, y, verbose=False, **fit_params)
        self._evals_result = self.model_.evals_result()
        return self

    def predict(self, X, **predict_params):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return pd.Series(self.model_.predict(X, **predict_params))

    def score(self, X, y, **kwargs):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.score(X, y, **kwargs)

    def get_metrics(self, X, y, metrics: List[BaseMetric] = None):
        y_pred = self.predict(X)
        y_pred_proba = (
            self.model_.predict_proba(X)
            if hasattr(self.model_, "predict_proba")
            else None
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
        if self._evals_result is not None:
            train_key = list(self._evals_result.keys())[0]
            metric_key = list(self._evals_result[train_key].keys())[0]
            return self._evals_result[train_key][metric_key]
        raise AttributeError("No loss curve available. Make sure fit() was called.")

    def get_loss_history_metric(self, X=None, y=None, metric: BaseMetric = None):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if metric:
            if X is None or y is None:
                raise ValueError(
                    "X and y must be provided to compute metric at each step."
                )
            n_estimators = self.model_.get_booster().num_boosted_rounds()
            results = []
            for i in range(1, n_estimators + 1):
                y_pred_proba = (
                    self.model_.predict_proba(X, iteration_range=(0, i))
                    if hasattr(self.model_, "predict_proba")
                    else None
                )
                y_pred = self.model_.predict(X, iteration_range=(0, i))
                try:
                    results.append(metric(y, y_pred_proba))
                except ValueError:
                    results.append(metric(y, y_pred))
            return results
        if self._evals_result is not None:
            train_key = list(self._evals_result.keys())[0]
            metric_key = list(self._evals_result[train_key].keys())[0]
            return self._evals_result[train_key][metric_key]
        raise AttributeError("No loss curve available. Make sure fit() was called.")

    @property
    def has_loss_history(self):
        return self._evals_result is not None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_class.__name__})"
