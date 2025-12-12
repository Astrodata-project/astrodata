import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, log_loss

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models.XGBoostModel import XGBoostModel


def _toy_classification(n_samples=80, n_features=6, random_state=123):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        n_classes=2,
        random_state=random_state,
    )
    return pd.DataFrame(X), pd.Series(y)


def test_xgboost_classifier_fit_predict_and_history(tmp_path):
    xgb = pytest.importorskip("xgboost", reason="xgboost missing")
    X, y = _toy_classification()
    model = XGBoostModel(
        xgb.XGBClassifier, n_estimators=12, max_depth=2, use_label_encoder=False
    )
    model.fit(X, y)

    yhat = model.predict(X)
    assert isinstance(yhat, pd.Series)
    assert len(yhat) == len(y)

    # loss history available via evals_result
    assert model.has_loss_history is True
    hist = model.get_loss_history()
    assert len(hist) == 12

    # staged metric evolution
    res = model.get_loss_history_metrics(
        X, y, metrics=[SklearnMetric(accuracy_score), SklearnMetric(log_loss)]
    )
    assert "accuracy_score_step" in res
    assert len(res["accuracy_score_step"]) == 12

    # save/load roundtrip
    p = tmp_path / "xgb.joblib"
    model.save(p)
    model2 = XGBoostModel(xgb.XGBClassifier)
    model2.load(p)
    yhat2 = model2.predict(X)
    assert np.allclose(yhat.values, yhat2.values)


def test_xgboost_get_set_params():
    xgb = pytest.importorskip("xgboost", reason="xgboost missing")
    model = XGBoostModel(
        xgb.XGBClassifier, n_estimators=10, max_depth=3, use_label_encoder=False
    )
    params = model.get_params()
    assert params["n_estimators"] == 10
    assert params["max_depth"] == 3
    assert params["model_class"] == xgb.XGBClassifier

    model.set_params(n_estimators=20, max_depth=5)
    updated_params = model.get_params()
    assert updated_params["n_estimators"] == 20
    assert updated_params["max_depth"] == 5


def test_xgboost_clone():
    xgb = pytest.importorskip("xgboost", reason="xgboost missing")
    model = XGBoostModel(
        xgb.XGBClassifier, n_estimators=10, max_depth=3, use_label_encoder=False
    )
    model_clone = model.clone()
    assert model_clone is not model
    assert model_clone.model_class == model.model_class
    assert model_clone.model_params == model.model_params


def test_xgboost_score():
    xgb = pytest.importorskip("xgboost", reason="xgboost missing")
    X, y = _toy_classification()
    model = XGBoostModel(
        xgb.XGBClassifier, n_estimators=10, max_depth=2, use_label_encoder=False
    )
    model.fit(X, y)
    score = model.score(X, y)
    assert isinstance(score, float)
    assert 0 <= score <= 1  # For classifiers, score is typically accuracy


def test_xgboost_regressor():
    xgb = pytest.importorskip("xgboost", reason="xgboost missing")
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=80, n_features=6, random_state=123)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    model = XGBoostModel(xgb.XGBRegressor, n_estimators=10, max_depth=2)
    model.fit(X, y)

    yhat = model.predict(X)
    assert isinstance(yhat, pd.Series)
    assert len(yhat) == len(y)

    # Regressor should use r2_score as default scorer
    scorer = model.get_scorer_metric()
    assert scorer.get_name() == "r2_score"
