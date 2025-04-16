from astrodata.data.schemas import RawData
from astrodata.data.preprocessors.base import AbstractPreprocessor


class NormalizeAndSplit(AbstractPreprocessor):
    """
    A preprocessor that normalizes the data by subtracting the mean and dividing
    by the standard deviation.

    Methods:
        transform(raw: RawData) -> RawData:
            Normalizes the data in the `RawData` object and returns the updated object.
    """

    def transform(self, raw: RawData) -> RawData:
        raw.data = (raw.data - raw.data.mean()) / raw.data.std()
        return raw


class MissingValueImputer(AbstractPreprocessor):
    """
    A preprocessor that imputes missing values in the data by replacing them with 0.

    Methods:
        transform(raw: RawData) -> RawData:
            Fills missing values in the `RawData` object with 0 and returns the updated object.
    """

    def transform(self, raw: RawData) -> RawData:
        raw.data = raw.data.fillna(0)
        return raw


class DropDuplicates(AbstractPreprocessor):
    """
    A preprocessor that removes duplicate rows from the data.

    Methods:
        transform(raw: RawData) -> RawData:
            Drops duplicate rows in the `RawData` object and returns the updated object.
    """

    def transform(self, raw: RawData) -> RawData:
        raw.data = raw.data.drop_duplicates()
        return raw
