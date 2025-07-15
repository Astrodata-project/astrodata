import pandas as pd

from astrodata.data import ProcessedData
from astrodata.preml import OHE, MissingImputator, PremlPipeline, TrainTestSplitter


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

    # PremlPipeline needs to know the configuration for each processor.
    # Either you define the processors directly in the code,
    # or you can define them in a configuration file. While using the latter approach,
    # each block in the config file should be named after the processor class name.
    # In case both methods are used, the processors defined in the code will take precedence over those defined in the config file.

    # Define the processors
    # It is mandatory to define a TrainTestSplitter processor, which will split the data into training, testing, and optionally validation sets.
    # Along with the specific parameters for each processor, you can also specify the save path for the artifacts.

    tts = TrainTestSplitter(targets=["target"], test_size=0.2, random_state=42)

    ohe_processor = OHE(
        categorical_columns=["feature2"],
        numerical_columns=["feature1", "feature3"],
    )

    MissingImputator = MissingImputator(
        categorical_columns=["feature2"],
        numerical_columns=["feature1", "feature3"],
    )

    # Define the PremlPipeline with the processors and configuration path
    preml_pipeline = PremlPipeline(
        config_path="example_config.yml",
        processors=[tts, MissingImputator, ohe_processor],
    )

    # Let's run the pipeline with the dummy processed data
    preml_data = preml_pipeline.run(processed_data)

    print("--" * 30)
    print("Preml Pipeline ran successfully!")
    print(f"Preml training features shape:{preml_data.train_features.shape}")
    print(f"Preml training targets shape:{preml_data.train_targets.shape}")
    print("--" * 30)

    # We will now try to define processors' parameters in the config file.
    # blocks should be named after the processor class names.
    # Order is important, as the first processor will be the TrainTestSplitter.
    # For example, If we put the ohe block before the missing imputator block,
    # the OHE will be applied before the missing values are handled.
    # parameters of the config file need to account fot this.

    config_path = "example_config_params.yaml"

    preml_pipeline = PremlPipeline(config_path=config_path)

    preml_data = preml_pipeline.run(processed_data)

    print("Preml Pipeline ran successfully!")
    print(f"Preml training features shape:{preml_data.train_features.shape}")
    print(f"Preml training targets shape:{preml_data.train_targets.shape}")

    # You can dump the preml data into supervised ML format, which will return train and test features and targets.
    X_train, X_test, y_train, y_test = preml_data.dump_supervised_ML_format()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
