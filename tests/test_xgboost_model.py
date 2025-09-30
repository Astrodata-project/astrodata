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
    model = XGBoostModel(xgb.XGBClassifier, n_estimators=12, max_depth=2, use_label_encoder=False)
    model.fit(X, y)

    yhat = model.predict(X)
    assert isinstance(yhat, pd.Series)
    assert len(yhat) == len(y)

    # loss history available via evals_result
    assert model.has_loss_history is True
    hist = model.get_loss_history()
    assert len(hist) == 12

    # staged metric evolution
    res = model.get_loss_history_metrics(X, y, metrics=[SklearnMetric(accuracy_score), SklearnMetric(log_loss)])
    assert "accuracy_score_step" in res
    assert len(res["accuracy_score_step"]) == 12

    # save/load roundtrip
    p = tmp_path / "xgb.joblib"
    model.save(p)
    model2 = XGBoostModel(xgb.XGBClassifier)
    model2.load(p)
    yhat2 = model2.predict(X)
    assert np.allclose(yhat.values, yhat2.values)
