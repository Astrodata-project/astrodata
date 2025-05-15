import yaml
from astrodata.data import ProcessedData
from astrodata.preml import Premldata
from sklearn.model_selection import train_test_split


def read_config(config_path: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def convert_to_preml_data(data: ProcessedData, config: dict) -> Premldata:
    """
    Converts a ProcessedData object to a Premldata object.

    Args:
        data (ProcessedData): The processed data to convert.

    Returns:
        Premldata: The converted Premldata object.
    """
    targets = config.get("targets", [data.data.columns[-1]])
    features_df = data.data.drop(columns=targets)
    targets_df = data.data[targets]

    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", None)
    validation = config.get("validation", {}).get("enabled", False)

    if validation:
        val_size = config.get("validation", {}).get("size", False)
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
