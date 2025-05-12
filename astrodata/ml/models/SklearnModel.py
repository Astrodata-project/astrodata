from astrodata.ml.models.BaseModel import BaseModel
import joblib

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
        return self.model_.predict(X, **predict_params)

    def save(self, filepath, **kwargs):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Nothing to save.")
        joblib.dump(self.model_, filepath, **kwargs)

    def load(self, filepath, **kwargs):
        self.model_ = joblib.load(filepath, **kwargs)

    def get_params(self, **kwargs):
        # Return the model parameters for reproducibility/grid search etc.
        params = {'model_class': self.model_class}
        params.update(self.model_params)
        return params

    def set_params(self, **params):
        # Set/Update the hyperparameters
        if 'model_class' in params:
            self.model_class = params.pop('model_class')
        self.model_params.update(params)
        return self

    def score(self, X, y, **kwargs):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.score(X, y, **kwargs)
    
    def __repr__(self):
        params = ', '.join(f"{k}={v!r}" for k, v in self.model_params.items())
        return f"{self.__class__.__name__}(model_class={self.model_class.__name__}, {params})"

