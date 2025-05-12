from abc import ABC, abstractmethod
from astrodata.data.schemas import RawData


class AbstractPreprocessor(ABC):
    """
    An abstract base class for data processors.

    Subclasses must implement the `preprocess` method to define how
    the input `RawData` is preprocessed.

    Methods:
        preprocess(raw: RawData) -> RawData:
            Abstract method to preprocess the input `RawData` and return
            a new `RawData` object.
    """

    @abstractmethod
    def preprocess(self, raw: RawData) -> RawData:
        """preprocess the input RawData and returns a new RawData object."""
        pass
