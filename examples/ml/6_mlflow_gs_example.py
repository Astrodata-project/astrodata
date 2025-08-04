import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.model_selection import GridSearchCVSelector
from astrodata.ml.models import SklearnModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker

# This example demonstrates how to use the tracking capabilities of astrodata.ml with a GridSearchCVSelector.
# It performs hyperparameter tuning on a GradientBoostingClassifier model using cross-validation and tracks the
# results using MLflow.
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

    # Set up the MLflow tracker with run name, experiment name, and additional tags
    # the tracker will log the model training and evaluation metrics to MLflow.
    # by providing a tracking_uri, tracking_username and tracking_password, you can connect to a remote MLflow server.

    tracker = SklearnMLflowTracker(
        run_name="GridSearchCVRun",
        experiment_name="examples_ml_6_mlflow_gs_example.py",
        extra_tags=None,
    )

    # Define the metrics to be used for evaluation

    accuracy = SklearnMetric(accuracy_score)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss, greater_is_better=False)

    metrics = [accuracy, f1, logloss]

    # Create the GridSearchCVSelector with the model, parameter grid, and metrics
    # This time we add a tracker to log the model training and evaluation metrics to MLflow.
    # log_all_models=False means that only the best model will be uploaded to MLflow.

    gss = GridSearchCVSelector(
        gradientboost,
        cv=2,
        param_grid={
            "n_estimators": [50, 100],
            "learning_rate": [0.01],
            "max_depth": [3, 5],
        },
        scorer=logloss,
        tracker=tracker,
        log_all_models=False,
        metrics=metrics,
    )

    gss.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    print(f"Best parameters found: {gss.get_best_params()}")
    print(f"Best metrics: {gss.get_best_metrics()}")

    # Here we tag for production the best model found during the grid search. The experiments in mlflow
    # are organized by the specified metric and the best performing one is registered.
    # make sure to use the same metric as the one used as scorer in the GridSearchCVSelector.

    tracker.register_best_model(
        metric=logloss,
        split_name="val",
        stage="production",
    )
