from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from astrodata.ml.metrics.BaseMetric import BaseMetric


class BaseMlModel(ABC):
    """
    Abstract base class for machine learning models.
    Defines the standard interface expected from all models.
    """

    def __init__(self):
        """
        Initialize the base model.
        """
        pass

    @abstractmethod
    def fit(self, X: Any, y: Any, **kwargs) -> "BaseMlModel":
        """
        Fit the model to the training data.

        Parameters
        ----------
        X : Any
            Training data features.
        y : Any
            Training data targets.
        **kwargs :
            Additional fit options.

        Returns
        -------
        BaseMlModel
            Returns self.
        """
        pass

    @abstractmethod
    def predict(self, X: Any, **kwargs) -> Any:
        """
        Predict using the trained model.

        Parameters
        ----------
        X : Any
            Input data.
        **kwargs :
            Additional options for prediction.

        Returns
        -------
        Any
            Model predictions.
        """
        pass

    @abstractmethod
    def score(self, X: Any, y: Any, **kwargs) -> float:
        """
        Compute the model score on test data.

        Parameters
        ----------
        X : Any
            Test data features.
        y : Any
            Test data targets.
        **kwargs :
            Additional options for scoring.

        Returns
        -------
        float
            Model score.
        """
        pass

    @abstractmethod
    def get_scorer_metric(self) -> BaseMetric:
        """
        Returns the score function default metric.
        """
        pass

    @abstractmethod
    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the model to the given filepath.

        Parameters
        ----------
        filepath : str
            Path to save the model.
        **kwargs :
            Additional save options.
        """
        pass

    @abstractmethod
    def load(self, filepath: str, **kwargs) -> "BaseMlModel":
        """
        Load the model from the given filepath.

        Parameters
        ----------
        filepath : str
            Path to load the model from.
        **kwargs :
            Additional load options.

        Returns
        -------
        BaseMlModel
            The loaded model instance.
        """
        pass

    @abstractmethod
    def get_metrics(self, X_test: Any, y_test: Any, **kwargs) -> Dict[str, float]:
        """
        Compute and return model metrics on test data.

        Parameters
        ----------
        X_test : Any
            Test data features.
        y_test : Any
            Test data targets.
        **kwargs :
            Additional metrics options.

        Returns
        -------
        dict
            Dictionary of metric names and values.
        """
        pass

    def get_params(self, **kwargs) -> Dict[str, Any]:
        """
        Get hyperparameters of the model.

        Returns
        -------
        dict
            Model hyperparameters.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        """
        Set hyperparameters of the model.

        Parameters
        ----------
        **kwargs :
            Model hyperparameters.

        Raises
        ------
        NotImplementedError
            If not implemented by the subclass.
        """
        raise NotImplementedError

    def clone(self) -> "BaseMlModel":
        """
        Create a (shallow) clone of this model instance.

        Returns
        -------
        BaseMlModel
            Cloned model instance.
        """
        new_instance = self.__class__(model_class=self.model_class, **self.model_params)
        # Copy over any callable attributes (e.g., decorated methods)
        for attr, value in self.__dict__.items():
            if callable(value):
                setattr(new_instance, attr, value)
        return new_instance
