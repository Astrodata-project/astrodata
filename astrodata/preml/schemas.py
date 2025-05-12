from pydantic import BaseModel
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split


class Premldata(BaseModel):
    """
    Represents processed data after transformations.

    Attributes:
        data (pd.DataFrame): The actual data as a Pandas DataFrame.
        metadata (Optional[dict]): Additional metadata about the processed data.
    """

    train_features: pd.DataFrame
    val_features: Optional[pd.DataFrame] = None
    test_features: pd.DataFrame
    train_targets: pd.DataFrame | pd.Series
    val_targets: Optional[pd.DataFrame | pd.Series] = None
    test_targets: pd.DataFrame | pd.Series
    metadata: Optional[dict] = {}

    class Config:
        arbitrary_types_allowed = True
