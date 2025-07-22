from astrodata.preml import OHE, MissingImputator, PremlPipeline


def run_preml_example(config, processed, tracker):
    # This step demonstrates how to use the PremlPipeline to perform additional preprocessing steps
    # for machine learning tasks, such as handling missing values and encoding categorical features.
    # The PremlPipeline class orchestrates these steps, similar to the DataPipeline, but is focused
    # on preparing data for supervised ML tasks. You define a sequence of processors to apply to the data.
    # Here, we use a MissingImputator to handle missing values and an OHE (One Hot Encoder) to encode categorical features.

    # Define the OHE processor for categorical and numerical columns
    ohe_processor = OHE(
        categorical_columns=["PULocationID"],
        numerical_columns=["trip_distance"],
    )

    # Define the MissingImputator processor for categorical and numerical columns
    missingImputator = MissingImputator(
        categorical_columns=["PULocationID"],
        numerical_columns=["trip_distance"],
    )

    # Define the PremlPipeline with the processors and configuration
    preml_pipeline = PremlPipeline(config, [missingImputator, ohe_processor])

    # Run the pipeline with the processed data
    preml_data = preml_pipeline.run(processed)

    # Again, we track the processed dataset and the (optional) artifacts
    tracker.track("Preml pipeline run, preml data versioned")

    print("Preml Pipeline ran successfully!")
    print(f"Preml data shape:{preml_data.train_features.shape}")
    print(f"Preml data shape:{preml_data.train_targets.shape}")

    # Dump the preml data into supervised ML format (train/test features and targets)
    X_train, X_test, y_train, y_test = preml_data.dump_supervised_ML_format()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test
