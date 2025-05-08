import xgboost as xgb
from astrodata.models.BaseModel import BaseModel
import joblib

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def config(self, model, **kwargs):
        """
        Set up the XGBoost model.
        
        Args:
            **kwargs: Parameters to initialize XGBClassifier (e.g., max_depth, n_estimators).
        """
        self.model = model(**kwargs)

    def fit(self, X, y, **kwargs):
        """
        Train the XGBoost model.
        
        Args:
            X: Training features.
            y: Training targets.
            **kwargs: Additional parameters for XGBClassifier.fit.
        """
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        """
        Make predictions using the trained XGBoost model.
        
        Args:
            X: Input data for predictions.
            **kwargs: Additional parameters for XGBClassifier.predict.
        
        Returns:
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model not configured or loaded.")
        return self.model.predict(X, **kwargs)

    def save(self, filepath, **kwargs):
        """
        Save the XGBoost model to a file.
        
        Args:
            filepath: File path to save the model.
            **kwargs: Additional parameters for saving (unused).
        """
        if self.model is None:
            raise ValueError("Model not configured or loaded.")
        joblib.dump(self.model, filepath)

    def load(self, filepath, **kwargs):
        """
        Load the XGBoost model from a file.
        
        Args:
            filepath: File path to load the model from.
            **kwargs: Additional parameters for loading (unused).
        """
        self.model = joblib.load(filepath)
