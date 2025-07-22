# MLflow

The `astrodata.tracking` module provides capabilities for experiment tracking, primarily through integration with MLflow. This allows users to log model parameters, metrics, artifacts, and manage different versions of their machine learning models.

## Core Concepts

  * **`MlflowBaseTracker`**: This class provides the core functionalities for integrating with MLflow. It handles:
      * **MLflow Configuration**: Setting the tracking URI, experiment name, and authentication details.
      * **Run Management**: Starting and ending MLflow runs.
      * **Logging**: Logging parameters, metrics, artifacts (models).
      * **Model Registration**: Registering the best performing models in the MLflow Model Registry.
  * **`SklearnMLflowTracker` (as seen in example)**: This is a specialized `MlflowBaseTracker` tailored for scikit-learn models, providing convenience methods for logging and tracking.

## How to Use (`SklearnMLflowTracker` with `HyperOptSelector` example)

The [7_mlflow_hp_example](<project:../python_examples/ml/7_mlflow_hp_example.rst>) file demonstrates a full integration of `HyperOptSelector` with MLflow tracking.

```python
import pandas as pd
from hyperopt import hp
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.HyperOptSelector import HyperOptSelector
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker

# To check the results, you can use the MLflow UI by running `mlflow ui` in your terminal
# and navigating to http://localhost:5000 in your web browser.

if __name__ == "__main__":

    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Instantiate and configure the Sklearn model
    gradientboost = SklearnModel(model_class=GradientBoostingClassifier)

    # Initialize the MLflow tracker
    # Setting an experiment name helps organize your runs in the MLflow UI
    tracker = SklearnMLflowTracker(experiment_name="Hyperopt_GBM_Experiment")

    # Define the hyperopt search space
    param_space = {
        "model": hp.choice("model", [gradientboost]),
        "n_estimators": hp.choice("n_estimators", [50, 100]),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
        "max_depth": hp.choice("max_depth", [3, 5]),
    }

    # Instantiate HyperOptSelector with the tracker
    # log_all_models=False means that only the best model will be uploaded to MLflow.
    accuracy = SklearnMetric(metric=accuracy_score)
    hos = HyperOptSelector(
        param_space=param_space,
        scorer=accuracy,
        use_cv=False,
        random_state=42,
        max_evals=10,
        metrics=[accuracy], # Metrics to log
        tracker=tracker, # Pass the tracker here
        log_all_models=False
    )

    hos.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    print("Best parameters found: ", hos.get_best_params())
    print("Best metrics: ", hos.get_best_metrics())

    # After the fit, the best model will be automatically logged and can be registered.
    # Here we tag for production the best model found during the grid search.
    # The experiments in mlflow are organized by the specified metric and the best performing one is registered.
    tracker.register_best_model(metric=accuracy, registered_model_name="GradientBoostingClassifier_Best_Model", split_name="val", stage="Production")
```

This example shows how to:

1.  Initialize `SklearnMLflowTracker`.
2.  Pass the `tracker` instance to `HyperOptSelector`.
3.  The `fit` method of `HyperOptSelector` will then automatically log parameters and metrics to MLflow for each evaluation.
4.  Finally, `tracker.register_best_model` is used to register the best model found during the hyperparameter search into the MLflow Model Registry, making it easy to manage and deploy.