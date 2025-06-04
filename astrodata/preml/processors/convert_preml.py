from astrodata.preml.processors.base import AbstractProcessor
from astrodata.preml.schemas import Premldata
from astrodata.data.schemas import ProcessedData
from sklearn.model_selection import train_test_split


class ConvertToPremlData(AbstractProcessor):
    """
    Processor to convert ProcessedData to Premldata.
    """

    def __init__(self, config: dict):
        super().__init__()
        try:
            self.config = config["preml"]
        except KeyError:
            raise ValueError("Config does not contain 'test_train_split' section.")

        self.artifact = self.config

    def process(self, data: ProcessedData, artifact: str = None, **kwargs):
        """
        Converts a ProcessedData object to a Premldata object.
        """
        targets = self.config.get("targets", [data.data.columns[-1]])
        features_df = data.data.drop(columns=targets)
        targets_df = data.data[targets]

        test_size = self.config.get("test_size", 0.2)
        random_state = self.config.get("random_state", None)
        validation = self.config.get("validation", {}).get("enabled", False)

        if validation:
            val_size = self.config.get("validation", {}).get("size", False)
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
