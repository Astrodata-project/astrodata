from abc import ABC, abstractmethod
from typing import Any
from astrodata.data.schemas import RawData


class AbstractProcessor(ABC):
    """
    An abstract base class for data processors.

    Subclasses must implement the `process` method to define how
    the input `RawData` is processed.

    Methods:
        process(raw: RawData) -> RawData:
            Abstract method to process the input `RawData` and return
            a new `RawData` object.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the processor with an empty dictionary to store artifacts.
        """
        self.kwargs = kwargs

    @abstractmethod
    def process(self, raw: RawData) -> RawData:
        """process the input RawData and returns a new RawData object."""
        pass
