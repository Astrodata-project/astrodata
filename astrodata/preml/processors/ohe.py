from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from .base import AbstractProcessor
from astrodata.preml import Premldata

from typing import Optional


class OHE(AbstractProcessor):

    # TODO: Change this to include custom parameters in init

    def __init__(
        self,
        categorical_columns: Optional[list] = None,
        numerical_columns: Optional[list] = None,
    ):
        """
        Initializes the OHE processor with optional categorical columns.
        Args:
            categorical_columns (Optional[list]): List of categorical columns to encode.
        """
        super().__init__(
            categorical_columns=categorical_columns, numerical_columns=numerical_columns
        )

    def process(
        self,
        preml: Premldata,
        artifact: Optional[str] = None,
        categorical_columns: Optional[list] = None,
        numerical_columns: Optional[list] = None,
        save_path: Optional[str] = None,
    ) -> Premldata:
        """
        One-hot encodes categorical features in the data.
        Args:
            preml (Premldata): The data to be processed.
            artifact (Optional[str]): One hot encoding artifact.
            categorical_columns (Optional[list]): List of categorical columns to encode.
            numerical_columns (Optional[list]): List of numerical columns to retain.
        Returns:
            Premldata: The processed data with one-hot encoded features.
        """
        if artifact:
            self.load_artifact(artifact)
            cat_ohe = self.artifact.transform(preml.test_features[categorical_columns])
            ohe_df = pd.DataFrame(
                cat_ohe,
                columns=self.artifact.get_feature_names_out(categorical_columns),
                index=preml.test_features.index,
            )
            preml.test_features = pd.concat(
                [preml.test_features[numerical_columns], ohe_df], axis=1
            )
        else:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ohe.fit(preml.train_features[categorical_columns])
            cat_ohe = ohe.transform(preml.train_features[categorical_columns])
            ohe_df = pd.DataFrame(
                cat_ohe,
                columns=ohe.get_feature_names_out(categorical_columns),
                index=preml.train_features.index,
            )
            preml.train_features = pd.concat(
                [preml.train_features[numerical_columns], ohe_df], axis=1
            )
            self.save_artifact(save_path)

        return preml
