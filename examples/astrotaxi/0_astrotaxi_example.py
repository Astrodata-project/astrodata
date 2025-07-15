from step1_data_import import run_data_import_example
from step2_preml import run_preml_example
from step3_ml import run_hyperopt_example  


def run_astrotaxi_example():
    # Step 1: Data Import
    processed = run_data_import_example()

    # Step 2: Pre-ML Processing
    X_train, y_train, X_test, y_test = run_preml_example(processed)

    # Step 3: Hyperparameter Optimization with HyperOpt
    run_hyperopt_example(X_train, y_train, X_test, y_test)
    
if __name__ == "__main__":
    run_astrotaxi_example()
    print("AstroTaxi example completed successfully!")