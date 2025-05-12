from pydantic import BaseModel
import pandas as pd
from typing import Optional, Literal


class RawData(BaseModel):
    """
    Represents raw input data loaded from a source file.

    Attributes:
        source (str): The source of the data (e.g., file path or URL).
        format (Literal): The format of the data (e.g., "fits", "hdf5", "csv", "parquet").
        data (pd.DataFrame): The actual data as a Pandas DataFrame.
    """

    source: str
    format: Literal["fits", "hdf5", "csv", "parquet"]
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
