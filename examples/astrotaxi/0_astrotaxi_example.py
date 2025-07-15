from step1_data_import import run_data_import_example
from step2_preml import run_preml_example
from step3_ml import run_hyperopt_example

from astrodata.tracking.Tracker import Tracker


def run_astrotaxi_example():

    config = "./examples/astrotaxi/config.yaml"
    tracker = Tracker(config)

    # Step 1: Data Import
    processed = run_data_import_example(config, tracker)

    # Step 2: Pre-ML Processing
    X_train, y_train, X_test, y_test = run_preml_example(config, processed, tracker)

    # Step 3: Hyperparameter Optimization with HyperOpt
    run_hyperopt_example(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    run_astrotaxi_example()
    print("AstroTaxi example completed successfully!")
