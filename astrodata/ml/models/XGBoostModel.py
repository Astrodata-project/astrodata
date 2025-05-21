import joblib
import pandas as pd

from astrodata.ml.models.BaseModel import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, model_class, **model_params):
        """
        Args:
            model_class: XGBoost model class (XGBClassifier or XGBRegressor).
            **model_params: Parameters for the XGBoost model.
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
