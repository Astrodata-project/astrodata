from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel


class RawData(BaseModel):
    """
    Represents raw input data loaded from a source file.

    Attributes:
        source (str): The source of the data (e.g., file path or URL).
        format (str): The format of the data (e.g., 'csv', 'parquet').
        data (pd.DataFrame): The actual data as a Pandas DataFrame.
    """

    source: Path | str
    format: str
    data: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


class ProcessedData(BaseModel):
    """
    Represents processed data after transformations.

    Attributes:
        data (pd.DataFrame): The actual data as a Pandas DataFrame.
        metadata (Optional[dict]): Additional metadata about the processed data.
    """

    data: pd.DataFrame
    metadata: Optional[dict] = {}

    class Config:
        arbitrary_types_allowed = True

    def dump_parquet(self, path: Path):
        """
        Dumps the processed data to a Parquet file.

        Args:
            path (Path): The file path to save the Parquet file.
        """
        self.data.to_parquet(path, index=False)
