import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector import GridSearchCVSelector
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Instantiate and configure the XGBoost model
gradientboost = SklearnModel(model_class=GradientBoostingClassifier)

tracker = SklearnMLflowTracker(
    run_name="GridSearchCVRun",
    experiment_name="examples_7_sklearnboosting_tracking.py",
    extra_tags={"stage": "testing"},
)

accuracy = SklearnMetric(accuracy_score)
f1 = SklearnMetric(f1_score, average="micro")
logloss = SklearnMetric(log_loss, greater_is_better=False)


metrics = [accuracy, f1, logloss]

gss = GridSearchCVSelector(
    gradientboost,
    cv=2,
    param_grid={
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    },
    scorer=logloss,
    tracker=tracker,
    log_all_models=False,
    metrics=metrics,
)

gss.fit(X_train, y_train, X_test=X_test, y_test=y_test)

print(gss.get_best_params())
print(gss.get_best_metrics())

tracker.register_best_model(
    metric=logloss,
    split_name="test",
    stage="Production",
)
