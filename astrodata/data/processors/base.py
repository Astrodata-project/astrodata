from abc import ABC, abstractmethod

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

    @abstractmethod
    def process(self, raw: RawData) -> RawData:
        """process the input RawData and returns a new RawData object."""
        pass
