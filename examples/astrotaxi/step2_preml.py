from astrodata.preml import OHE, MissingImputator, PremlPipeline


def run_preml_example(processed):
    config_path = "./testdata/green_tripdata_2024-01_config.yaml"

    ohe_processor = OHE(
        # categorical_columns=["PULocationID", "DOLocationID"],
        # numerical_columns=["passenger_count", "trip_distance", "duration"],
        categorical_columns=["PULocationID"],
        numerical_columns=["trip_distance"],
        save_path="./testdata/ohe.pkl",
    )

    missingImputator = MissingImputator(
        categorical_columns=["PULocationID"],
        numerical_columns=["trip_distance"],
        save_path="./testdata/imputer.pkl",
    )

    preml_pipeline = PremlPipeline(config_path, [missingImputator, ohe_processor])

    preml_data = preml_pipeline.run(processed)

    print("Preml Pipeline ran successfully!")
    print(f"Preml data shape:{preml_data.train_features.shape}")
    print(f"Preml data shape:{preml_data.train_targets.shape}")

    X_train, X_test, y_train, y_test = preml_data.dump_supervised_ML_format()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test
