from abc import ABC, abstractmethod
from typing import Any

from astrodata.ml.models import BaseMlModel


class ModelTracker(ABC):
    """
    Abstract base class for tracking model fitting processes.
    """

    @abstractmethod
    def wrap_fit(self, obj: BaseMlModel) -> BaseMlModel:
        """
        Wrap the fit method of an object to add tracking or logging.

        Parameters
        ----------
        obj : Any
            The object whose fit method will be wrapped.

        Returns
        -------
        BaseMlModel
            The wrapped object.
        """
        pass
