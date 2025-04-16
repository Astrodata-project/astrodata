from pydantic import BaseModel
import pandas as pd
from typing import Any, Optional, Literal


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
        features (Any): The features extracted or transformed from the raw data.
        labels (Optional[Any]): The labels corresponding to the features, if applicable.
        metadata (Optional[dict]): Additional metadata about the processed data.
    """

    features: Any
    labels: Optional[Any]
    metadata: Optional[dict] = {}
