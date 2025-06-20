import pytest
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector import GridSearchSelector, GridSearchCVSelector
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.models.XGBoostModel import XGBoostModel

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@pytest.fixture(scope="module")
def regression_data():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return train_test_split(X, y, test_size=0.25, random_state=42)

@pytest.fixture(scope="module")
def metrics():
    return [
        SklearnMetric(mean_absolute_error),
        SklearnMetric(mean_squared_error),
        SklearnMetric(r2_score),
    ]

@pytest.fixture(scope="module")
def param_grids():
    return {
        "skl": {
            "model": SklearnModel(model_class=GradientBoostingRegressor),
            "param_grid": {
                "n_estimators": [50],
                "learning_rate": [0.1],
                "max_depth": [3],
            },
        },
        "xgb": {
            "model": XGBoostModel(model_class=XGBRegressor, tree_method="hist", enable_categorical=True),
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
def test_regression_model_selection(regression_data, metrics, param_grids, selector_cls, model_key):
    X_train, X_test, y_train, y_test = regression_data
    model = param_grids[model_key]["model"]
    param_grid = param_grids[model_key]["param_grid"]
    scorer = metrics[1]  # MSE
    
    if selector_cls is GridSearchSelector:
        selector = selector_cls(model, val_size=0.2, param_grid=param_grid, scorer=scorer, metrics=metrics)
    else:
        selector = selector_cls(model, cv=2, param_grid=param_grid, scorer=scorer, metrics=metrics)
    
    selector.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=False)
    best_params = selector.get_best_params()
    best_metrics = selector.get_best_metrics()
    
    # Assert that best_params and best_metrics are not empty
    assert best_params, "No best parameters found"
    assert best_metrics, "No best metrics found"
