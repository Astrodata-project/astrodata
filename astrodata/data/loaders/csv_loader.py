import pandas as pd

from astrodata.data.loaders.base import BaseLoader
from astrodata.data.schemas import RawData


class CsvLoader(BaseLoader):
    """
    Data loader for CSV files.

    This class implements the `BaseLoader` interface to load data
    from CSV files using pandas.

    Methods:
        load(path: str) -> RawData:
            Loads data from the specified CSV file path and returns
            it as a `RawData` object.
    """

    def load(self, path: str) -> RawData:
        return RawData(source=path, format="csv", data=pd.read_csv(path))
