from abc import ABC, abstractmethod
from astrodata.data.schemas import RawData


class AbstractPreprocessor(ABC):
    """
    An abstract base class for data preprocessors.

    Subclasses must implement the `transform` method to define how
    the input `RawData` is transformed.

    Methods:
        transform(raw: RawData) -> RawData:
            Abstract method to transform the input `RawData` and return
            a new `RawData` object.
    """

    @abstractmethod
    def transform(self, raw: RawData) -> RawData:
        """Transforms the input RawData and returns a new RawData object."""
        pass
