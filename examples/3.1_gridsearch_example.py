import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from astrodata.ml.metrics.classification import Accuracy
from astrodata.ml.model_selection.GridSearchSelector import GridSearchSelector
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

model = SklearnModel(model_class=RandomForestClassifier, random_state=42)
# model = XGBoostModel(model_class=XGBClassifier, tree_method="hist", enable_categorical=True)

tracker = SklearnMLflowTracker(
    log_model=True,
    run_name="DemoRun",
    experiment_name="examples/3.1_gridsearch_example.py",
    extra_tags={"stage": "testing"},
)

param_grid = {"n_estimators": [10, 50, 100], "max_depth": [None, 2, 3]}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = tracker.wrap_fit(
    model=model,
    X_test=X_test,
    y_test=y_test,
    input_example=X_train.iloc[:5],
    metrics=[Accuracy()],
)

selector = GridSearchSelector(
    model, param_grid, val_size=0.2, random_state=42, metrics=[Accuracy()]
)

selector.fit(X, y)
print("Best params:", selector.get_best_params())
print("Best metrics:", selector.get_best_metrics())
