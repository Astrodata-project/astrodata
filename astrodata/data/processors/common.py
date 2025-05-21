from astrodata.data.processors.base import AbstractProcessor
from astrodata.data.schemas import RawData


class NormalizeAndSplit(AbstractProcessor):
    """
    A processor that normalizes the data by subtracting the mean and dividing
    by the standard deviation.

    Methods:
        process(raw: RawData) -> RawData:
            Normalizes the data in the `RawData` object and returns the updated object.
    """

    def process(self, raw: RawData) -> RawData:
        raw.data = (raw.data - raw.data.mean()) / raw.data.std()
        return raw


class MissingValueImputer(AbstractProcessor):
    """
    A processor that imputes missing values in the data by replacing them with 0.

    Methods:
        process(raw: RawData) -> RawData:
            Fills missing values in the `RawData` object with 0 and returns the updated object.
    """

    def process(self, raw: RawData) -> RawData:
        raw.data = raw.data.fillna(0)
        return raw


class DropDuplicates(AbstractProcessor):
    """
    A processor that removes duplicate rows from the data.

    Methods:
        process(raw: RawData) -> RawData:
            Drops duplicate rows in the `RawData` object and returns the updated object.
    """

    def process(self, raw: RawData) -> RawData:
        raw.data = raw.data.drop_duplicates()
        return raw
