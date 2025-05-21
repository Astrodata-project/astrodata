import pandas as pd

from astrodata.data.loaders.base import BaseLoader
from astrodata.data.schemas import RawData


class ParquetLoader(BaseLoader):
    """
    Data loader for Parquet files.

    This class implements the `BaseLoader` interface to load data
    from Parquet files using pandas.

    Methods:
        load(path: str) -> RawData:
            Loads data from the specified Parquet file path and returns
            it as a `RawData` object.
    """

    def load(self, path: str) -> RawData:
        return RawData(source=path, format="parquet", data=pd.read_parquet(path))
