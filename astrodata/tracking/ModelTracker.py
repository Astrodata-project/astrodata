from abc import ABC, abstractmethod
from typing import Any

from astrodata.ml.models import BaseModel


class ModelTracker(ABC):
    """
    Abstract base class for tracking model fitting processes.
    """

    @abstractmethod
    def wrap_fit(self, obj: BaseModel) -> BaseModel:
        """
        Wrap the fit method of an object to add tracking or logging.

        Parameters
        ----------
        obj : Any
            The object whose fit method will be wrapped.

        Returns
        -------
        BaseModel
            The wrapped object.
        """
        pass
