import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from astrodata.ml.metrics.SklearnMetric import SklearnMetric


def test_sklearn_metric_basic_usage():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    acc = SklearnMetric(accuracy_score)
    assert acc.get_name() == "accuracy_score"
    assert acc.greater_is_better is True
    assert acc(y_true, y_pred) == accuracy_score(y_true, y_pred)


def test_sklearn_metric_custom_name_and_kwargs():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    mse = SklearnMetric(mean_squared_error, name="mse", greater_is_better=False)
    assert mse.get_name() == "mse"
    assert mse.greater_is_better is False
    assert mse(y_true, y_pred) == mean_squared_error(y_true, y_pred)

