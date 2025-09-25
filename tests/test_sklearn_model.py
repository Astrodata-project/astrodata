import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models.SklearnModel import SklearnModel


def _toy_classification(n_samples=60, n_features=4, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=random_state,
    )
    return pd.DataFrame(X), pd.Series(y)


def test_fit_predict_score_logistic_regression():
    X, y = _toy_classification()
    m = SklearnModel(LogisticRegression, solver="liblinear", random_state=0)

    with pytest.raises(RuntimeError):
        m.predict(X)

    m.fit(X, y)
    yhat = m.predict(X)
    assert isinstance(yhat, pd.Series)
    assert len(yhat) == len(y)

    proba = m.predict_proba(X)
    assert hasattr(proba, "shape") and proba.shape[1] == 2

    score = m.score(X, y)
    assert isinstance(score, float)


def test_get_metrics_and_default_scorer():
    X, y = _toy_classification()
    m = SklearnModel(LogisticRegression, solver="liblinear", random_state=0)
    m.fit(X, y)
    # Default scorer for classifiers should be accuracy
    scorer = m.get_scorer_metric()
    assert isinstance(scorer, SklearnMetric)
    assert scorer.get_name() == "accuracy_score"

    metrics = [SklearnMetric(accuracy_score), SklearnMetric(log_loss)]
    res = m.get_metrics(X, y, metrics=metrics)
    assert "accuracy_score" in res
    assert "log_loss" in res


def test_loss_history_with_gradient_boosting():
    X, y = _toy_classification()
    m = SklearnModel(GradientBoostingClassifier, random_state=0, n_estimators=10)
    m.fit(X, y)
    assert m.has_loss_history is True
    hist = m.get_loss_history()
    assert hasattr(hist, "__len__") and len(hist) == 10

    # staged metrics
    res = m.get_loss_history_metrics(X, y, metrics=[SklearnMetric(accuracy_score)])
    assert "accuracy_score_step" in res
    assert len(res["accuracy_score_step"]) == 10


def test_save_and_load_roundtrip(tmp_path):
    X, y = _toy_classification()
    m = SklearnModel(LogisticRegression, solver="liblinear", random_state=0)
    m.fit(X, y)
    path = tmp_path / "model.joblib"
    m.save(path)

    m2 = SklearnModel(LogisticRegression, solver="liblinear")
    m2.load(path)
    yhat1 = m.predict(X)
    yhat2 = m2.predict(X)
    assert np.allclose(yhat1.values, yhat2.values)

