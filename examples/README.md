# Examples README

This README provides an overview of the example scripts included in the `examples` directory. These examples demonstrate how to use different components of the `astrodata` library for data preprocessing, model training, hyperparameter optimization, and experiment tracking. **Note:** Refer to the main `astrodata` documentation for a complete description of the package.

## Table of Contents

- [Examples README](#examples-readme)
  - [Table of Contents](#table-of-contents)
  - [1. Preprocessing and PremlPipeline](#1-preprocessing-and-premlpipeline)
  - [2. Basic Model Training with SklearnModel](#2-basic-model-training-with-sklearnmodel)
  - [3. Multiple Models in a Loop](#3-multiple-models-in-a-loop)
  - [4. Grid Search Hyperparameter Tuning](#4-grid-search-hyperparameter-tuning)
  - [5. Hyperopt Hyperparameter Tuning](#5-hyperopt-hyperparameter-tuning)
  - [6. MLflow Tracking Examples](#6-mlflow-tracking-examples)
    - [a. Simple MLflow Tracking](#a-simple-mlflow-tracking)
    - [b. Grid Search with MLflow Tracking](#b-grid-search-with-mlflow-tracking)
    - [c. HyperOpt with MLflow Tracking](#c-hyperopt-with-mlflow-tracking)
  - [7. Configuration Files](#7-configuration-files)
  - [Requirements \& Notes](#requirements--notes)
  - [Quick Start](#quick-start)
  - [License](#license)


## 1. Preprocessing and PremlPipeline

**File:** `data/2_preml_example.py`  
**Config Example:** `example_config_params.yaml`

- Demonstrates how to use `PremlPipeline` for preprocessing tasks.
- Utilizes processors such as `OHE` (One Hot Encoder), `MissingImputator`, and `TrainTestSplitter`.
- Processors can be defined either programmatically or via a YAML configuration file.
- Shows how to handle missing values, encode categorical features, and split data for machine learning tasks.
- Outputs the shapes of resulting train/test splits and demonstrates dumping data into a supervised ML format.

**Config sample (`example_config_params.yaml`):**
```yaml
preml:
  TrainTestSplitter:
    targets: ["target"]
    test_size: 0.2
    random_state: 42
  MissingImputator:
    categorical_columns: ["feature2"]
    numerical_columns: ["feature1", "feature3"]
  OHE:
    categorical_columns: ["feature2"]
    numerical_columns: ["feature1", "feature3"]
```



## 2. Basic Model Training with SklearnModel

**File:** `ml/1_sklearn_example.py`

- Shows how to wrap a scikit-learn model (`LinearSVR`) using `SklearnModel`.
- Demonstrates how to define and use metrics (`mean_absolute_error`, `mean_squared_error`, `r2_score`, etc.) via `SklearnMetric`.
- Trains and evaluates a model on the diabetes dataset.



## 3. Multiple Models in a Loop

**File:** `ml/2_multimodel_example.py`

- Illustrates training and evaluating multiple models (`SklearnModel` and `XGBoostModel`) in a single loop.
- Uses the breast cancer dataset as an example.
- Compares metrics (accuracy, F1-score) across different model types.



## 4. Grid Search Hyperparameter Tuning

**File:** `ml/3_gridsearch_example.py`

- Demonstrates hyperparameter tuning with `GridSearchCVSelector`.
- Uses `SklearnModel` and scikit-learn's `LinearSVC`.
- Specifies parameter grids and scorer functions.
- Finds and prints the best parameters and model metrics.



## 5. Hyperopt Hyperparameter Tuning

**File:** `ml/4_hyperopt_example.py`

- Shows how to use `HyperOptSelector` for hyperparameter optimization.
- Defines the search space using the `hyperopt` library.
- Evaluates metrics such as accuracy, F1-score, and log loss.
- Prints the best parameters and corresponding model metrics.



## 6. MLflow Tracking Examples

### a. Simple MLflow Tracking

**File:** `ml/5_mlflow_simple_example.py`

- Demonstrates how to use `SklearnMLflowTracker` to track model training and evaluation.
- Logs metrics and models to MLflow.
- To view results, run `mlflow ui` and navigate to `http://localhost:5000`.

### b. Grid Search with MLflow Tracking

**File:** `ml/6_mlflow_gs_example.py`

- Shows how to combine `GridSearchCVSelector` with MLflow tracking.
- Hyperparameter search is logged and the best model can be registered for production.

### c. HyperOpt with MLflow Tracking

**File:** `ml/7_mlflow_hp_example.py`

- Demonstrates hyperparameter optimization with `HyperOptSelector` and MLflow tracking.
- Best models and experiments are logged and can be tagged for production.

---

## 7. Configuration Files

- **`example_config.yaml`**: Sample configuration for train/test splitting.
- **`example_config_params.yaml`**: Extended config for specifying parameters for all pipeline processors.

---

## Requirements & Notes

- All examples require the `astrodata` package and its dependencies (`scikit-learn`, `xgboost`, `hyperopt`, `mlflow`, etc.).
- Some examples require an MLflow server to be running for tracking.
- For more information or troubleshooting, refer to the main `astrodata` [documentation](https://astrodata-project.github.io/astrodata/).

---

## Quick Start

Install dependencies (if not already):

```bash
pip install git+https://astrodata-project.github.io/astrodata/
```

Run any example:

```bash
python 1_sklearn_example.py
```

---

## License

See the main `astrodata` repository for license information.