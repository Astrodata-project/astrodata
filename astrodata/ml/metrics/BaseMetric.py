from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """
    Abstract base class for metric objects.

    Subclasses must implement initialization, call, name retrieval,
    and indicate if greater values are better.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the metric object.
        """
        pass

    @abstractmethod
    def __call__(self, y_true: Any, y_pred: Any, **kwargs) -> float:
        """
        Compute the metric.

        Parameters
        ----------
        y_true : Any
            Ground truth target values.
        y_pred : Any
            Predicted values.
        **kwargs :
            Additional keyword arguments for the metric computation.

        Returns
        -------
        float
            The computed metric value.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the metric.

        Returns
        -------
        str
            The metric name.
        """
        pass

    @property
    @abstractmethod
    def greater_is_better(self) -> bool:
        """
        Whether higher metric values are better.

        Returns
        -------
        bool
            True if greater values are better, False otherwise.
        """
        pass

    def __eq__(self, other: object) -> bool:
        """
        Compare this metric object with another based on metric name.

        Parameters
        ----------
        other : object
            Another metric object.

        Returns
        -------
        bool
            True if metric names are equal, False otherwise.
        """
        if hasattr(other, "get_name"):
            return self.get_name() == other.get_name()
        return NotImplemented
