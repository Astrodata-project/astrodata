from typing import Optional

from sklearn.preprocessing import StandardScaler

from astrodata.preml.schemas import Premldata

from .base import PremlProcessor


class Standardizer(PremlProcessor):
    """
    Standardizer for scaling numerical features.

    This class provides functionality to standardize numerical columns in a dataset
    by scaling them to have a mean of 0 and a standard deviation of 1. It supports
    saving and loading scaling artifacts for reuse.
    """

    def __init__(
        self,
        numerical_columns: Optional[list] = None,
        artifact_path: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Initializes the Standardizer with optional column specifications.

        Args:
            numerical_columns (Optional[list]): List of numerical columns to standardize.
            save_path (Optional[str]): Path to save the scaling artifact.
        """
        super().__init__(
            artifact_path=artifact_path,
            numerical_columns=numerical_columns,
            save_path=save_path,
        )

    def process(
        self,
        preml: Premldata,
    ) -> Premldata:
        """
        Standardizes numerical features in the dataset.

        This method scales numerical columns to have a mean of 0 and a standard deviation of 1.
        If an artifact path is provided, it loads the scaling artifact and applies it to the test
        features. Otherwise, it fits a new scaler on the training features, transforms both training
        and test features, and saves the artifact for reuse.

        Args:
            preml (Premldata): The data to be processed.

        Returns:
            Premldata: The processed data with standardized numerical features.
        """
        if self.artifact:
            scaler = self.artifact

            preml.test_features[self.kwargs["numerical_columns"]] = scaler.transform(
                preml.test_features[self.kwargs["numerical_columns"]]
            )
            if hasattr(preml, "val_features") and preml.val_features is not None:
                preml.val_features[self.kwargs["numerical_columns"]] = scaler.transform(
                    preml.val_features[self.kwargs["numerical_columns"]]
                )
        else:
            # Standardize numerical columns
            scaler = StandardScaler()
            scaler.fit(preml.train_features[self.kwargs["numerical_columns"]])
            preml.train_features[self.kwargs["numerical_columns"]] = scaler.transform(
                preml.train_features[self.kwargs["numerical_columns"]]
            )

            self.save_artifact(scaler)

            # Apply scaler to test features
            preml.test_features[self.kwargs["numerical_columns"]] = scaler.transform(
                preml.test_features[self.kwargs["numerical_columns"]]
            )
            if hasattr(preml, "val_features") and preml.val_features is not None:
                preml.val_features[self.kwargs["numerical_columns"]] = scaler.transform(
                    preml.val_features[self.kwargs["numerical_columns"]]
                )
        return preml
