from sklearn.model_selection import train_test_split

from astrodata.data.schemas import ProcessedData
from astrodata.preml.processors.base import PremlProcessor
from astrodata.preml.schemas import Premldata


class TrainTestSplitter(PremlProcessor):
    """
    Processor to convert ProcessedData to Premldata.

    This processor splits the input ProcessedData into training, testing, and optionally validation sets
    according to the configuration provided. It supports specifying target columns, test size, random state,
    and validation split. The output is a Premldata object containing the split datasets and metadata.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.artifact = kwargs

    def process(self, data: ProcessedData, **kwargs):
        """
        Converts a ProcessedData object to a Premldata object.

        This method splits the input ProcessedData into training, testing, and optionally validation sets
        using scikit-learn's train_test_split. The configuration determines the target columns, test size,
        random state, and validation split. The resulting Premldata object contains the split features,
        targets, and metadata.

        Args:
            data (ProcessedData): The input processed data to be split.
        Returns:
            Premldata: The resulting Premldata object containing the split datasets.
        """
        targets = self.kwargs.get("targets", [data.data.columns[-1]])
        features_df = data.data.drop(columns=targets)
        targets_df = data.data[targets]

        test_size = self.kwargs.get("test_size", 0.2)
        random_state = self.kwargs.get("random_state", None)
        validation = self.kwargs.get("validation", {}).get("enabled", False)

        if validation:
            val_size = self.kwargs.get("validation", {}).get("size", False)
            X_temp, X_test, y_temp, y_test = train_test_split(
                features_df, targets_df, test_size=test_size, random_state=random_state
            )
            val_relative_size = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_relative_size, random_state=random_state
            )
            return Premldata(
                train_features=X_train,
                val_features=X_val,
                test_features=X_test,
                train_targets=y_train,
                val_targets=y_val,
                test_targets=y_test,
                metadata=data.metadata,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets_df, test_size=test_size, random_state=random_state
            )
            return Premldata(
                train_features=X_train,
                test_features=X_test,
                train_targets=y_train,
                test_targets=y_test,
                metadata=data.metadata,
            )
