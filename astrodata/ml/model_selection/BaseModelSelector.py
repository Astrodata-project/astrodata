from abc import ABC, abstractmethod
from typing import Any, Dict

from astrodata.ml.models import BaseModel


class BaseModelSelector(ABC):
    """
    Abstract base class for model selection strategies.

    Subclasses must implement methods for fitting the selector to data,
    retrieving the best model, and retrieving the best hyperparameters.
    """

    def __init__(self):
        """Initialize the BaseModelSelector."""
        pass

    @abstractmethod
    def fit(self, X: Any, y: Any, *args, **kwargs) -> "BaseModelSelector":
        """
        Fit the model selector to data.

        Parameters
        ----------
        X : Any
            Training data features.
        y : Any
            Training data targets.
        *args, **kwargs :
            Additional arguments for fitting.

        Returns
        -------
        BaseModelSelector
            Returns self.
        """
        pass

    @abstractmethod
    def get_best_model(self) -> BaseModel:
        """
        Return the best model found during selection.

        Returns
        -------
        Any
            The best model object.
        """
        pass

    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """
        Return the best hyperparameters found during selection.

        Returns
        -------
        dict
            Dictionary of best parameters.
        """
        pass

    def get_params(self, **kwargs) -> Dict[str, Any]:
        """
        Return parameters of the selector. Can be optionally overridden.

        Returns
        -------
        dict
            Dictionary of selector parameters.
        """
        pass
