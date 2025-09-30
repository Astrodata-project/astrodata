import pytest
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import log_loss, mean_squared_error

from astrodata.ml.metrics._utils import get_loss_func


def test_get_loss_func_for_sklearn_loss_param():
    # SGDRegressor exposes a 'loss' param. Default is 'squared_error'.
    model = SGDRegressor(loss="squared_error")
    f = get_loss_func(model)
    assert f is mean_squared_error


def test_get_loss_func_for_unknown_returns_none():
    model = object()
    assert get_loss_func(model) is None


def test_get_loss_func_for_xgboost_binary():
    xgb = pytest.importorskip("xgboost")
    model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=5)
    f = get_loss_func(model)
    assert f is log_loss

