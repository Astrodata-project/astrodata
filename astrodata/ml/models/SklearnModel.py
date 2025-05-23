from typing import List

import joblib
import pandas as pd

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseModel import BaseModel


class SklearnModel(BaseModel):
    def __init__(self, model_class, **model_params):
        """
        Args:
            model_class: The sklearn estimator class (not an instance!).
            **model_params: Hyperparameters for the sklearn model.
        """
        self.model_class = model_class
        self.model_params = model_params
        self.model_ = None

    def fit(self, X, y, **fit_params):
        self.model_ = self.model_class(**self.model_params)
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return pd.Series(self.model_.predict(X, **predict_params))

    def save(self, filepath, **kwargs):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Nothing to save.")
        joblib.dump(self.model_, filepath, **kwargs)

    def load(self, filepath, **kwargs):
        self.model_ = joblib.load(filepath, **kwargs)

    def get_params(self, **kwargs):
        # Return the model parameters for reproducibility/grid search etc.
        params = {"model_class": self.model_class}
        params.update(self.model_params)
        return params

    def set_params(self, **params):
        # Set/Update the hyperparameters
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

    def get_train_score(self):
        """
        Returns the train_score_ if available, else raises an error.
        This is typically available for sklearn's boosting models.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if hasattr(self.model_, "train_score_"):
            # train_score_ is an array of the loss at each iteration
            return self.model_.train_score_
        else:
            raise AttributeError(
                f"The fitted model of type {type(self.model_)} does not have a 'train_score_' attribute. "
                "Loss curves are typically available for boosting models like GradientBoostingClassifier/Regressor."
            )

    @property
    def has_train_score(self):
        """Returns True if the fitted model exposes a train_score_ attribute."""
        return self.model_ is not None and hasattr(self.model_, "train_score_")
