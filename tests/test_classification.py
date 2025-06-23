import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector import (
    GridSearchCVSelector,
    GridSearchSelector,
)
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.models.XGBoostModel import XGBoostModel


@pytest.fixture(scope="module")
def classification_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return train_test_split(X, y, test_size=0.25, random_state=42)


@pytest.fixture(scope="module")
def metrics():
    return [
        SklearnMetric(accuracy_score),
        SklearnMetric(f1_score, average="micro"),
        SklearnMetric(log_loss, greater_is_better=False),
    ]


@pytest.fixture(scope="module")
def param_grids():
    return {
        "skl": {
            "model": SklearnModel(model_class=GradientBoostingClassifier),
            "param_grid": {
                "n_estimators": [50],
                "learning_rate": [0.1],
                "max_depth": [3],
            },
        },
        "xgb": {
            "model": XGBoostModel(
                model_class=XGBClassifier, tree_method="hist", enable_categorical=True
            ),
            "param_grid": {
                "n_estimators": [50],
                "learning_rate": [0.1],
                "max_depth": [3],
                "colsample_bytree": [1.0],
            },
        },
    }


@pytest.mark.parametrize("selector_cls", [GridSearchSelector, GridSearchCVSelector])
@pytest.mark.parametrize("model_key", ["skl", "xgb"])
def test_classification_model_selection(
    classification_data, metrics, param_grids, selector_cls, model_key
):
    X_train, X_test, y_train, y_test = classification_data
    model = param_grids[model_key]["model"]
    param_grid = param_grids[model_key]["param_grid"]
    scorer = metrics[2]  # logloss

    if selector_cls is GridSearchSelector:
        selector = selector_cls(
            model, val_size=0.2, param_grid=param_grid, scorer=scorer, metrics=metrics
        )
    else:
        selector = selector_cls(
            model, cv=2, param_grid=param_grid, scorer=scorer, metrics=metrics
        )

    selector.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=False)
    best_params = selector.get_best_params()
    best_metrics = selector.get_best_metrics()
    history_metrics = selector.get_best_model().get_loss_history_metrics(
        X_test, y_test, metrics
    )

    # Assert that best_params and best_metrics are not empty
    assert best_params, "No best parameters found"
    assert best_metrics, "No best metrics found"
    assert isinstance(history_metrics, dict), "No best history_metrics found"
