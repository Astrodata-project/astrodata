import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import SklearnModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker

# This example demonstrates how to use the tracking capabilities of astrodata.ml with a simple model.
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
        run_name="MlFlowSimpleRun",
        experiment_name="examples_ml_5_mlflow_simple_example.py",
        extra_tags={"stage": "testing"},
    )

    # Define the metrics to be used for evaluation

    accuracy = SklearnMetric(accuracy_score)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss, greater_is_better=False)

    metrics = [accuracy, f1, logloss]

    # By using the tracker function wrap fit the model "gains" the ability to track itself on the specified mlflow server.
    # X_test and y_test are passed as arguments to be used in metric tracking. After the model is wrapped, you can use the same
    # methods that the wrapped model had before (fit, predict, get_metrics, ...).

    tracked_gradientboost = tracker.wrap_fit(
        gradientboost,
        X_test=X_test,
        y_test=y_test,
        metrics=metrics,
        log_model=True,
    )

    tracked_gradientboost.fit(X_train, y_train)

    print(
        "Metrics on test set:",
        tracked_gradientboost.get_metrics(
            X_test,
            y_test,
            metrics=metrics,
        ),
    )
