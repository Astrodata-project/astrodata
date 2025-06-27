from typing import Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from astrodata.preml.schemas import Premldata

from .base import AbstractProcessor


class OHE(AbstractProcessor):
    """
    OneHotEncoder (OHE) processor for encoding categorical features.

    This class provides functionality to one-hot encode categorical columns
    in a dataset, while optionally retaining numerical columns. It supports
    saving and loading encoding artifacts using pickle for reuse.
    """

    def __init__(
        self,
        categorical_columns: Optional[list] = None,
        numerical_columns: Optional[list] = None,
        artifact: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Initializes the OHE processor with optional categorical columns.
        Args:
            categorical_columns (Optional[list]): List of categorical columns to encode.
            numerical_columns (Optional[list]): List of numerical columns to retain.
            save_path (Optional[str]): Path to save the encoding artifact.
        """
        super().__init__(
            artifact=artifact,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            save_path=save_path,
        )

    def process(
        self,
        preml: Premldata,
    ) -> Premldata:
        """
        One-hot encodes categorical features in the data.

        This method encodes categorical columns in the dataset using one-hot encoding.
        If an artifact path is provided, it loads the encoding artifact and applies it
        to the test features. Otherwise, it fits a new encoder on the training features,
        transforms both training and test features, and saves the artifact for reuse.

        Args:
            preml (Premldata): The data to be processed.
            artifact (Optional[str]): Path to a saved one-hot encoding artifact.

        Returns:
            Premldata: The processed data with one-hot encoded features.
        """
        if self.artifact:
            cat_ohe = self.artifact.transform(
                preml.test_features[self.kwargs["categorical_columns"]]
            )
            ohe_df = pd.DataFrame(
                cat_ohe,
                columns=self.artifact.get_feature_names_out(
                    self.kwargs["categorical_columns"]
                ),
                index=preml.test_features.index,
            )
            preml.test_features = pd.concat(
                [preml.test_features[self.kwargs["numerical_columns"]], ohe_df], axis=1
            )
        else:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ohe.fit(preml.train_features[self.kwargs["categorical_columns"]])
            cat_ohe_train = ohe.transform(
                preml.train_features[self.kwargs["categorical_columns"]]
            )
            ohe_df_train = pd.DataFrame(
                cat_ohe_train,
                columns=ohe.get_feature_names_out(self.kwargs["categorical_columns"]),
                index=preml.train_features.index,
            )
            preml.train_features = pd.concat(
                [preml.train_features[self.kwargs["numerical_columns"]], ohe_df_train],
                axis=1,
            )
            if self.kwargs.get("save_path"):
                self.save_artifact(ohe, self.kwargs["save_path"])
            cat_ohe_test = ohe.transform(
                preml.test_features[self.kwargs["categorical_columns"]]
            )
            ohe_df_test = pd.DataFrame(
                cat_ohe_test,
                columns=ohe.get_feature_names_out(self.kwargs["categorical_columns"]),
                index=preml.test_features.index,
            )
            preml.test_features = pd.concat(
                [preml.test_features[self.kwargs["numerical_columns"]], ohe_df_test],
                axis=1,
            )

        return preml
