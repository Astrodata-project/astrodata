from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from astrodata.models.BaseModel import BaseModel
import joblib

class SklearnModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None  # Will hold the sklearn estimator

    def config(self, model, **kwargs):
        """
        Set up the sklearn model.
        
        Args:
            model: An instance of an sklearn estimator (e.g., LinearRegression()).
            **kwargs: Optional parameters to set on the model.
        """
        self.model = model
        

    def fit(self, X, y, **kwargs):
        """
        Train the model.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            **kwargs: Additional fit parameters.
        """
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """
        Predict using the model.
        
        Args:
            X (np.ndarray): Feature matrix.
            **kwargs: Additional predict parameters.
        """
        return self.model.predict(X, **kwargs)

    def save(self, filepath, **kwargs):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model.
        """
        if self.model is None:
            raise RuntimeError("Model is not configured. Nothing to save.")
        joblib.dump(self.model, filepath, **kwargs)

    def load(self, filepath, **kwargs):
        """
        Load the model from a file.
        
        Args:
            filepath (str): Path to load the model from.
        """
        self.model = joblib.load(filepath, **kwargs)
