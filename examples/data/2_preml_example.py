import pandas as pd

from astrodata.data import ProcessedData
from astrodata.preml import OHE, MissingImputator, Premldata, PremlPipeline


def dummy_processed_data():
    # Create a dummy DataFrame with some missing values and categorical data
    data = {
        "feature1": [1, 2, None, 4],
        "feature2": ["A", "B", "A", None],
        "feature3": [10.5, 20.5, 30.5, 40.5],
        "target": [0, 1, 0, 1],
    }
    return ProcessedData(data=pd.DataFrame(data))


if __name__ == "__main__":
    # This example demonstrates how to perform additional preprocessing steps that involves machine learning tasks.
    # We will use a One Hot Encoder (OHE) to encode categorical features and a Missing Imputator to handle missing values.
    # The PremlPipeline class orchestrates these preprocessing steps, allowing us to prepare the data for machine learning tasks.
    # Its concept is similar to the DataPipeline, where you define a sequence of processors to apply to the data.
    # We will create a dummy processed DataFrame, apply OHE and MissingImputator, and print the results.
    processed_data = dummy_processed_data()

    # PremlPipeline needs a configuration file path, where you must define parameters for train_test_split.
    # Optionally, you can define blocks for each processor, where you can specify parameters for each processor.
    config_path = "example_config.yaml"

    # Define the processors
    # Along with the specific parameters for each processor, you can also specify the save path for the artifacts.
    ohe_processor = OHE(
        categorical_columns=["feature2"],
        numerical_columns=["feature1", "feature3"],
    )

    missing_imputator = MissingImputator(
        categorical_columns=["feature2"],
        numerical_columns=["feature1", "feature3"],
    )

    # Define the PremlPipeline with the processors and configuration path
    preml_pipeline = PremlPipeline([missing_imputator, ohe_processor], config_path)

    # Let's run the pipeline with the dummy processed data
    preml_data = preml_pipeline.run(processed_data)

    # We will now try to define processors' parameters in the config file.
    # blocks should be named after the processor class names.
    config_path = "example_config_params.yaml"
    ohe_processor = OHE()
    missing_imputator = MissingImputator()

    preml_pipeline = PremlPipeline([missing_imputator, ohe_processor], config_path)

    preml_data = preml_pipeline.run(processed_data)

    print("Preml Pipeline ran successfully!")
    print(f"Preml data shape:{preml_data.train_features.shape}")
    print(f"Preml data shape:{preml_data.train_targets.shape}")

    # You can dump the preml data into supervised ML format, which will return train and test features and targets.
    X_train, X_test, y_train, y_test = preml_data.dump_supervised_ML_format()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
