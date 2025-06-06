import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector import GridSearchSelector
from astrodata.ml.models.XGBoostModel import XGBoostModel
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
gradientboost = XGBoostModel(model_class=XGBClassifier)

tracker = SklearnMLflowTracker(
    log_model=True,
    run_name="DemoRun",
    experiment_name="examples/8_xgboosting_tracking.py",
    extra_tags={"stage": "testing"},
)

metrics = [SklearnMetric(accuracy_score), SklearnMetric(f1_score, average="micro")]

tracked_gradientboost = tracker.wrap_fit(
    model=gradientboost,
    X_test=X,
    y_test=y,
    input_example=X_train.iloc[:5],
    metrics=metrics,
)

gss = GridSearchSelector(
    tracked_gradientboost,
    param_grid={
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    },
)

gss.fit(X_train, y_train)
print(gss.get_best_params())
print(gss.get_best_model())
