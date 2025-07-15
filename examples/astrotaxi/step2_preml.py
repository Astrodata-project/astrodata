from astrodata.preml import OHE, MissingImputator, PremlPipeline


def run_preml_example(config, processed, tracker):

    ohe_processor = OHE(
        categorical_columns=["PULocationID"],
        numerical_columns=["trip_distance"],
    )

    missingImputator = MissingImputator(
        categorical_columns=["PULocationID"],
        numerical_columns=["trip_distance"],
    )

    preml_pipeline = PremlPipeline(config, [missingImputator, ohe_processor])

    preml_data = preml_pipeline.run(processed)
    tracker.track("Preml pipeline run, preml data versioned")

    print("Preml Pipeline ran successfully!")
    print(f"Preml data shape:{preml_data.train_features.shape}")
    print(f"Preml data shape:{preml_data.train_targets.shape}")

    X_train, X_test, y_train, y_test = preml_data.dump_supervised_ML_format()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test
