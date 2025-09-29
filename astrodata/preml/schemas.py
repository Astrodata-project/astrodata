from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel


class Premldata(BaseModel):
    """
    Represents processed data after transformations.

    Attributes:
        train_features (pd.DataFrame): Training features.
        val_features (Optional[pd.DataFrame]): Validation features, if available.
        test_features (pd.DataFrame): Test features.
        train_targets (pd.DataFrame | pd.Series): Training targets.
        val_targets (Optional[pd.DataFrame | pd.Series]): Validation targets, if available.
        test_targets (pd.DataFrame | pd.Series): Test targets.
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

    def dump_parquet(self, path: Path):
        """
        Dumps the processed data to a Parquet file.

        Args:
            path (Path): The file path to save the Parquet file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.train_features.to_parquet(path / "train_features.parquet", index=False)
        if self.val_features is not None:
            self.val_features.to_parquet(path / "val_features.parquet", index=False)
        self.test_features.to_parquet(path / "test_features.parquet", index=False)
        self.train_targets.to_parquet(path / "train_targets.parquet", index=False)
        if self.val_targets is not None:
            self.val_targets.to_parquet(path / "val_targets.parquet", index=False)
        self.test_targets.to_parquet(path / "test_targets.parquet", index=False)
