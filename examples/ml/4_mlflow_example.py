import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")

tracker = SklearnMLflowTracker(
    log_model=True,
    run_name="sklearn_run",
    experiment_name="DemoExperiment",
    extra_tags={"stage": "testing"},
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = tracker.wrap_fit(
    model=model,
    X_test=X_test,
    y_test=y_test,
    input_example=X_train.iloc[:5],
    metrics=[
        SklearnMetric(accuracy_score),
        SklearnMetric(f1_score, average="macro"),
        SklearnMetric(confusion_matrix),
    ],
)

model.fit(X_train, y_train)
