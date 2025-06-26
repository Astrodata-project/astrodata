from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_clusterer, is_outlier_detector, is_regressor
from sklearn.metrics import accuracy_score, adjusted_rand_score, r2_score, roc_auc_score

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel


class SklearnModel(BaseMlModel):
    """
    A wrapper class for scikit-learn models to standardize the interface and add extended functionality.
    """

    def __init__(self, model_class, **model_params):
        """
        Initialize the SklearnModel.

        Parameters
        ----------
        model_class : type
            The scikit-learn model class.
        **model_params : dict
            Parameters for the scikit-learn model.
        """
        self.model_class = model_class
        self.model_params = model_params
        self.model_ = None

    def get_params(self, **kwargs) -> dict:
        """
        Get parameters for this model.

        Returns
        -------
        dict
            Dictionary containing the model class and its parameters.
        """
        params = {"model_class": self.model_class}
        params.update(self.model_params)
        return params

    def set_params(self, **params) -> BaseMlModel:
        """
        Set parameters for this model.

        Parameters
        ----------
        **params : dict
            Parameters to set for the model.

        Returns
        -------
        self
            The updated SklearnModel instance.
        """
        if "model_class" in params:
            self.model_class = params.pop("model_class")
        self.model_params.update(params)
        return self

    def save(self, filepath, **kwargs) -> None:
        """
        Save the fitted model to a file.

        Parameters
        ----------
        filepath : str
            Path to file where the model will be saved.
        **kwargs
            Additional arguments passed to joblib.dump.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")
        joblib.dump(self.model_, filepath, **kwargs)

    def load(self, filepath, **kwargs) -> None:
        """
        Load a model from file.

        Parameters
        ----------
        filepath : str
            Path to file where the model is stored.
        **kwargs
            Additional arguments passed to joblib.load.
        """
        self.model_ = joblib.load(filepath, **kwargs)

    def fit(self, X, y, **fit_params) -> BaseMlModel:
        """
        Fit the model to data.

        Parameters
        ----------
        X : array-like
            Training data features.
        y : array-like
            Training data targets.
        **fit_params
            Additional fit parameters for the underlying model.

        Returns
        -------
        self
            The fitted SklearnModel instance.
        """
        self.model_ = self.model_class(**self.model_params)
        self.model_.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params) -> pd.Series:
        """
        Predict targets for samples in X.

        Parameters
        ----------
        X : array-like
            Input features.
        **predict_params
            Additional parameters for the underlying model's predict method.

        Returns
        -------
        pandas.Series
            Predicted values.

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return pd.Series(self.model_.predict(X, **predict_params))

    def predict_proba(self, X, **predict_params) -> pd.DataFrame:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like
            Input features.
        **predict_params
            Additional parameters for the underlying model's predict_proba method.

        Returns
        -------
        pandas.DataFrame
            Predicted probabilities.

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        AttributeError
            If the model does not support predict_proba.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if hasattr(self.model_, "predict_proba"):
            return pd.DataFrame(self.model_.predict_proba(X, **predict_params))
        raise AttributeError(f"{type(self.model_)} does not support predict_proba.")

    def score(self, X, y, **kwargs) -> float:
        """
        Return the score of the model on the given test data and labels.

        Parameters
        ----------
        X : array-like
            Test data features.
        y : array-like
            True labels for X.
        **kwargs
            Additional arguments for the underlying model's score method.

        Returns
        -------
        float
            Score of the model.

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model_.score(X, y, **kwargs)

    def get_scorer_metric(self) -> SklearnMetric:
        """
        Returns the default metric used by the model's scoring function.
        """
        match True:
            case _ if is_classifier(self.model_class()):
                return SklearnMetric(accuracy_score)
            case _ if is_regressor(self.model_class()):
                return SklearnMetric(r2_score)
            case _ if is_clusterer(self.model_class()):
                return SklearnMetric(adjusted_rand_score)
            case _ if is_outlier_detector(self.model_class()):
                return SklearnMetric(roc_auc_score)
            case _:
                raise ValueError("Unknown model type")

    def get_metrics(self, X, y, metrics: List[BaseMetric] = None) -> Dict[str, Any]:
        """
        Compute metrics for the given data.

        Parameters
        ----------
        X : array-like
            Data features.
        y : array-like
            Data targets.
        metrics : list of BaseMetric
            List of metrics to compute.

        Returns
        -------
        dict
            Mapping from metric names to metric values.
        """
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

    def get_loss_history(self) -> np.ndarray:
        """
        Get the loss history during training, if available.

        Returns
        -------
        array-like
            Loss or score history.

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        AttributeError
            If the model does not have 'train_score_' attribute.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if hasattr(self.model_, "train_score_"):
            return self.model_.train_score_
        raise AttributeError(f"{type(self.model_)} does not have 'train_score_'.")

    def get_loss_history_metrics(
        self, X=None, y=None, metrics: List[BaseMetric] = None
    ) -> dict:
        """
        Get the evolution of a metric over training stages.

        Parameters
        ----------
        X : array-like, optional
            Data features to use for staged predictions.
        y : array-like, optional
            True labels for X.
        metrics : List[BaseMetric], optional
            Metrics to compute for each stage.

        Returns
        -------
        list
            List of metric values for each training stage.

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        ValueError
            If X or y are not provided.
        AttributeError
            If the model does not support staged predictions.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")

        results = {}

        if self.has_loss_history:
            for metric in metrics:
                if X is None or y is None:
                    raise ValueError("X and y required for metric history.")
                try:
                    results[f"{metric.get_name()}_step"] = [
                        metric(y, y_pred)
                        for y_pred in self.model_.staged_predict_proba(X)
                    ]

                except (AttributeError, ValueError):
                    results[f"{metric.get_name()}_step"] = [
                        metric(y, y_pred) for y_pred in self.model_.staged_predict(X)
                    ]
        else:
            raise AttributeError(f"{type(self.model_)} does not support loss history.")
        return results

    @property
    def has_loss_history(self) -> bool:
        """
        Check if the underlying model supports loss history.

        Returns
        -------
        bool
            True if loss history is available, False otherwise.
        """
        return self.model_ is not None and hasattr(self.model_, "train_score_")

    def __repr__(self) -> str:
        """
        Return string representation of the SklearnModel.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"{self.__class__.__name__}({self.model_class.__name__})"
