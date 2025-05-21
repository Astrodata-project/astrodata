from typing import Optional

import pandas as pd
from pydantic import BaseModel


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

    def dump_supervised_ML_format(self):
        """
        Returns the data into training and testing sets.

        Returns:
            tuple: A tuple containing the training and testing features and targets.
        """
        if self.val_features is not None and self.val_targets is not None:
            return (
                self.train_features,
                self.val_features,
                self.test_features,
                self.train_targets,
                self.val_targets,
                self.test_targets,
            )
        return (
            self.train_features,
            self.test_features,
            self.train_targets,
            self.test_targets,
        )
