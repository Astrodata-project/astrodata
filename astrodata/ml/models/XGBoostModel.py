from typing import Any, Dict, List

import joblib
import pandas as pd

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel


class XGBoostModel(BaseMlModel):
    """
    Wrapper for XGBoost models, providing a standardized interface and additional utilities.
    """

    def __init__(self, model_class, **model_params):
        """
        Initialize the XGBoostModel.

        If no eval_metric is provided, uses "logloss" for classifiers and "rmse" otherwise.

        Parameters
        ----------
        model_class : type
            The XGBoost model class.
        **model_params : dict
            Hyperparameters for the XGBoost model.
        """
        if "eval_metric" not in model_params:
            model_params["eval_metric"] = (
                "logloss" if hasattr(model_class, "predict_proba") else "rmse"
            )
        self.model_class = model_class
        self.model_params = model_params
        self.model_ = None
        self._evals_result = None

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
        self : XGBoostModel
            The updated model instance.
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
            Path to the file where the model will be saved.
        **kwargs
            Additional arguments passed to joblib.dump.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
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
        Fit the XGBoost model.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training targets.
        **fit_params
            Additional parameters to pass to model.fit(). If "eval_set"
            is not provided, it will default to the training data.

        Returns
        -------
        self : XGBoostModel
            Fitted model.
        """
        if "eval_set" not in fit_params:
            fit_params["eval_set"] = [(X, y)]
        self.model_ = self.model_class(**self.model_params)
        self.model_.fit(X, y, verbose=False, **fit_params)
        self._evals_result = self.model_.evals_result()
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

    def get_loss_history(self) -> List[float]:
        """
        Get the training loss history from the last fit.

        Returns
        -------
        list or array-like
            Loss history for the first metric in the evals_result.

        Raises
        ------
        AttributeError
            If no loss history is available.
        """
        if self._evals_result is not None:
            train_key = list(self._evals_result.keys())[0]
            metric_key = list(self._evals_result[train_key].keys())[0]
            return self._evals_result[train_key][metric_key]
        raise AttributeError("No loss curve available. Make sure fit() was called.")

    def get_loss_history_metrics(
        self, X=None, y=None, metrics: List[BaseMetric] = None
    ) -> List[float]:
        """
        Get the evolution of a custom metric over boosting rounds.

        Parameters
        ----------
        X : array-like, optional
            Data features to use for staged predictions.
        y : array-like, optional
            True labels for X.
        metrics : List[BaseMetric], optional
            Metrics to compute for each boosting stage.

        Returns
        -------
        list
            List of metric values for each boosting round.

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        ValueError
            If X or y are not provided when a metric is requested.
        AttributeError
            If no loss history is available.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")

        results = {}

        if self.has_loss_history:
            for metric in metrics:
                if X is None or y is None:
                    raise ValueError(
                        "X and y must be provided to compute metric at each step."
                    )
                n_estimators = self.model_.get_booster().num_boosted_rounds()
                r = []
                for i in range(1, n_estimators + 1):
                    y_pred_proba = (
                        self.model_.predict_proba(X, iteration_range=(0, i))
                        if hasattr(self.model_, "predict_proba")
                        else None
                    )
                    y_pred = self.model_.predict(X, iteration_range=(0, i))
                    try:
                        r.append(metric(y, y_pred_proba))
                    except ValueError:
                        r.append(metric(y, y_pred))
                results[f"{metric.get_name()}_step"] = r

        else:
            raise AttributeError(f"{type(self.model_)} does not support loss history.")
        return results

    @property
    def has_loss_history(self) -> bool:
        """
        Check if loss history is available from the last fit.

        Returns
        -------
        bool
            True if loss history is available, False otherwise.
        """
        return self._evals_result is not None

    def __repr__(self) -> str:
        """
        Return string representation of the XGBoostModel.

        Returns
        -------
        str
            String representation of the object.
        """
        return f"{self.__class__.__name__}({self.model_class.__name__})"
