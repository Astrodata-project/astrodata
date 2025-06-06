import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector import GridSearchSelector
from astrodata.ml.models.SklearnModel import SklearnModel


@pytest.fixture
def diabetes_data():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


@pytest.fixture
def train_test_data(diabetes_data):
    X, y = diabetes_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_sklearn_model_fit_predict(train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    model = SklearnModel(model_class=LinearSVR, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert isinstance(preds, (pd.Series, pd.DataFrame, list))


def test_sklearn_model_metrics(train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    model = SklearnModel(model_class=LinearSVR, random_state=42)
    model.fit(X_train, y_train)
    metrics = model.get_metrics(
        X_test,
        y_test,
        metrics=[
            SklearnMetric(mean_absolute_error),
            SklearnMetric(mean_squared_error),
            SklearnMetric(r2_score),
            SklearnMetric(mean_squared_log_error),
        ],
    )
    assert isinstance(metrics, dict)
    assert "mean_absolute_error" in metrics
    assert "mean_squared_error" in metrics
    assert "r2_score" in metrics
    assert "mean_squared_log_error" in metrics
    # Check that metrics are floats
    for value in metrics.values():
        assert isinstance(value, float)


def test_sklearn_grid_search(train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    model = SklearnModel(model_class=LinearSVR, random_state=42)

    param_grid = {"C": [0.1, 1, 10], "epsilon": [0.1, 0.2]}

    selector = GridSearchSelector(
        model,
        param_grid,
        val_size=0.2,
        random_state=42,
        metrics=[SklearnMetric(mean_absolute_error)],
    )

    selector.fit(X_train, y_train)
    best_params = selector.get_best_params()
    best_model = selector.get_best_model()
    assert isinstance(best_model, SklearnModel)
    assert isinstance(best_params, dict)
    assert "C" in best_params
    assert "epsilon" in best_params
