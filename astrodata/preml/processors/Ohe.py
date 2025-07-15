from typing import Optional

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from astrodata.preml.schemas import Premldata

from .base import PremlProcessor


class OHE(PremlProcessor):
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
        artifact_path: Optional[str] = None,
    ):
        """
        Initializes the OHE processor with optional categorical columns.
        Args:
            artifact_path (Optional[str]): Path to a saved one-hot encoding artifact.
            categorical_columns (Optional[list]): List of categorical columns to encode.
            numerical_columns (Optional[list]): List of numerical columns to retain.
        """
        super().__init__(
            artifact_path=artifact_path,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
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

        def _transform_features(
            features, encoder, categorical_columns, numerical_columns
        ):
            cat_ohe = encoder.transform(features[categorical_columns])
            ohe_df = pd.DataFrame(
                cat_ohe,
                columns=encoder.get_feature_names_out(categorical_columns),
                index=features.index,
            )
            return pd.concat([features[numerical_columns], ohe_df], axis=1)

        if self.artifact:
            preml.test_features = _transform_features(
                preml.test_features,
                self.artifact,
                self.kwargs["categorical_columns"],
                self.kwargs["numerical_columns"],
            )
            if hasattr(preml, "val_features") and preml.val_features is not None:
                preml.val_features = _transform_features(
                    preml.val_features,
                    self.artifact,
                    self.kwargs["categorical_columns"],
                    self.kwargs["numerical_columns"],
                )
        else:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ohe.fit(preml.train_features[self.kwargs["categorical_columns"]])
            preml.train_features = _transform_features(
                preml.train_features,
                ohe,
                self.kwargs["categorical_columns"],
                self.kwargs["numerical_columns"],
            )
            self.save_artifact(ohe)
            preml.test_features = _transform_features(
                preml.test_features,
                ohe,
                self.kwargs["categorical_columns"],
                self.kwargs["numerical_columns"],
            )
            if hasattr(preml, "val_features") and preml.val_features is not None:
                preml.val_features = _transform_features(
                    preml.val_features,
                    ohe,
                    self.kwargs["categorical_columns"],
                    self.kwargs["numerical_columns"],
                )

        return preml
