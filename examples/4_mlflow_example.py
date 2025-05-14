from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker
from astrodata.ml.metrics.Accuracy import AccuracyMetric
from astrodata.ml.metrics.F1Metric import F1Metric
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer
import pandas as pd


data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")

tracker = SklearnMLflowTracker(
    log_model=True,
    run_name="sklearn_run",
    experiment_name="DemoExperiment",
    extra_tags={"stage": "testing"}
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = tracker.wrap_fit(model=model, X_test=X_test, y_test=y_test, input_example=X_train.iloc[:5], metric_classes=[AccuracyMetric,F1Metric])

model.fit(X_train, y_train)