from abc import ABC, abstractmethod
from astrodata.data.schemas import RawData


class AbstractTransformer(ABC):
    """
    An abstract base class for data transformers.

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


class NormalizeAndSplit(AbstractTransformer):
    """
    A transformer that normalizes the data by subtracting the mean and dividing
    by the standard deviation.

    Methods:
        transform(raw: RawData) -> RawData:
            Normalizes the data in the `RawData` object and returns the updated object.
    """

    def transform(self, raw: RawData) -> RawData:
        raw.data = (raw.data - raw.data.mean()) / raw.data.std()
        return raw


class MissingValueImputer(AbstractTransformer):
    """
    A transformer that imputes missing values in the data by replacing them with 0.

    Methods:
        transform(raw: RawData) -> RawData:
            Fills missing values in the `RawData` object with 0 and returns the updated object.
    """

    def transform(self, raw: RawData) -> RawData:
        raw.data = raw.data.fillna(0)
        return raw


class DropDuplicates(AbstractTransformer):
    """
    A transformer that removes duplicate rows from the data.

    Methods:
        transform(raw: RawData) -> RawData:
            Drops duplicate rows in the `RawData` object and returns the updated object.
    """

    def transform(self, raw: RawData) -> RawData:
        raw.data = raw.data.drop_duplicates()
        return raw
