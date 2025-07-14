from typing import Optional

from sklearn.impute import SimpleImputer

from astrodata.preml.schemas import Premldata

from .base import PremlProcessor


class MissingImputator(PremlProcessor):
    """
    Missing value imputator for handling missing data in datasets.

    This class provides functionality to impute missing values in numerical
    and categorical columns using specified strategies. It supports saving
    and loading imputation artifacts for reuse.
    """

    def __init__(
        self,
        categorical_columns: Optional[list] = None,
        numerical_columns: Optional[list] = None,
        artifact: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        Initializes the MissingImputator with optional column specifications.

        Args:
            categorical_columns (Optional[list]): List of categorical columns to impute.
            numerical_columns (Optional[list]): List of numerical columns to impute.
            save_path (Optional[str]): Path to save the imputation artifact.
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
        artifact: Optional[str] = None,
    ) -> Premldata:
        """
        Imputes missing values in the dataset.

        This method imputes missing values in numerical columns using the mean
        and in categorical columns using the mode. If an artifact path is provided,
        it loads the imputation artifact and applies it to the test features.
        Otherwise, it fits new imputers on the training features, transforms both
        training and test features, and saves the artifact for reuse.

        Args:
            preml (Premldata): The data to be processed.
            artifact (Optional[str]): Path to a saved imputation artifact.

        Returns:
            Premldata: The processed data with imputed values.
        """
        if artifact:
            self.load_artifact(artifact)
            num_imputer, cat_imputer = self.artifact

            preml.test_features[self.kwargs["numerical_columns"]] = (
                num_imputer.transform(
                    preml.test_features[self.kwargs["numerical_columns"]]
                )
            )
            preml.test_features[self.kwargs["categorical_columns"]] = (
                cat_imputer.transform(
                    preml.test_features[self.kwargs["categorical_columns"]]
                )
            )
            if hasattr(preml, "val_features") and preml.val_features is not None:
                preml.val_features[self.kwargs["numerical_columns"]] = (
                    num_imputer.transform(
                        preml.val_features[self.kwargs["numerical_columns"]]
                    )
                )
                preml.val_features[self.kwargs["categorical_columns"]] = (
                    cat_imputer.transform(
                        preml.val_features[self.kwargs["categorical_columns"]]
                    )
                )
        else:
            # Impute numerical columns with mean
            num_imputer = SimpleImputer(strategy="mean")
            num_imputer.fit(preml.train_features[self.kwargs["numerical_columns"]])
            preml.train_features[self.kwargs["numerical_columns"]] = (
                num_imputer.transform(
                    preml.train_features[self.kwargs["numerical_columns"]]
                )
            )

            # Impute categorical columns with mode
            cat_imputer = SimpleImputer(strategy="most_frequent")
            cat_imputer.fit(preml.train_features[self.kwargs["categorical_columns"]])
            preml.train_features[self.kwargs["categorical_columns"]] = (
                cat_imputer.transform(
                    preml.train_features[self.kwargs["categorical_columns"]]
                )
            )

            if self.kwargs.get("save_path"):
                self.save_artifact((num_imputer, cat_imputer), self.kwargs["save_path"])

            # Apply imputers to test features
            preml.test_features[self.kwargs["numerical_columns"]] = (
                num_imputer.transform(
                    preml.test_features[self.kwargs["numerical_columns"]]
                )
            )
            preml.test_features[self.kwargs["categorical_columns"]] = (
                cat_imputer.transform(
                    preml.test_features[self.kwargs["categorical_columns"]]
                )
            )
            if hasattr(preml, "val_features") and preml.val_features is not None:
                preml.val_features[self.kwargs["numerical_columns"]] = (
                    num_imputer.transform(
                        preml.val_features[self.kwargs["numerical_columns"]]
                    )
                )
                preml.val_features[self.kwargs["categorical_columns"]] = (
                    cat_imputer.transform(
                        preml.val_features[self.kwargs["categorical_columns"]]
                    )
                )

        return preml
