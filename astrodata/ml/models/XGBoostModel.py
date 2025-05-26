from typing import List

import joblib
import pandas as pd

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseModel import BaseModel

5


class XGBoostModel(BaseModel):
    def __init__(self, model_class, **model_params):
        """
        Args:
            model_class: XGBoost model class (XGBClassifier or XGBRegressor).
            **model_params: Parameters for the XGBoost model.
        """
        if "eval_metric" not in model_params:
            model_params["eval_metric"] = (
                "logloss" if hasattr(model_class, "predict_proba") else "rmse"
            )
        self.model_class = model_class
        self.model_params = model_params
        self.model_ = None

    def fit(self, X, y, **fit_params):
        if "eval_set" not in fit_params:
            fit_params["eval_set"] = [(X, y)]
        self.model_ = self.model_class(**self.model_params)
        self.model_.fit(X, y, **fit_params)
        self._evals_result = self.model_.evals_result()
        return self

    def predict(self, X, **predict_params):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return pd.Series(self.model_.predict(X, **predict_params))

    def save(self, filepath, **kwargs):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Nothing to save.")
        # XGBoost models have their own save method, but joblib works as well
        joblib.dump(self.model_, filepath, **kwargs)

    def load(self, filepath, **kwargs):
        self.model_ = joblib.load(filepath, **kwargs)

    def get_params(self, **kwargs):
        params = {"model_class": self.model_class}
        params.update(self.model_params)
        return params

    def set_params(self, **params):
        if "model_class" in params:
            self.model_class = params.pop("model_class")
        self.model_params.update(params)
        return self

    def score(self, X, y, **kwargs):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.score(X, y, **kwargs)

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.model_params.items())
        return f"{self.__class__.__name__}(model_class={self.model_class.__name__}, {params})"

    def get_metrics(self, X_test, y_test, metrics: List[BaseMetric] = None):
        y_pred = self.predict(X_test)
        results = {}
        for metric in metrics:
            score = metric(y_test, y_pred)
            results[metric.get_name()] = score
        return results

    def get_loss_history(self):
        """
        Returns the training loss curve if available, else raises an error.
        """
        if self._evals_result is not None:
            # Training set is always first in eval_set
            train_key = list(self._evals_result.keys())[0]
            metric_key = list(self._evals_result[train_key].keys())[0]
            return self._evals_result[train_key][metric_key]
        else:
            raise AttributeError("No loss curve available. Make sure fit() was called.")

    @property
    def has_loss_history(self):
        return self._evals_result is not None
