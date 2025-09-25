import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector import (
    GridSearchCVSelector,
    GridSearchSelector,
)
from astrodata.ml.models.SklearnModel import SklearnModel


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


def test_grid_search_selector_with_val_split():
    X, y = _toy_classification()
    model = SklearnModel(LogisticRegression, solver="liblinear", random_state=0)
    selector = GridSearchSelector(
        model=model,
        param_grid={"C": [0.1, 1.0]},
        scorer=SklearnMetric(accuracy_score),
        val_size=0.25,
        random_state=0,
        metrics=[SklearnMetric(accuracy_score)],
    )

    selector.fit(X, y)
    best_model = selector.get_best_model()
    best_params = selector.get_best_params()
    best_metrics = selector.get_best_metrics()

    assert best_model is not None
    assert isinstance(best_params, dict) and "C" in best_params
    assert isinstance(best_metrics, dict) and "accuracy_score" in best_metrics
    # ensure the model can predict
    yhat = best_model.predict(X)
    assert len(yhat) == len(y)


def test_grid_search_cv_selector():
    X, y = _toy_classification()
    model = SklearnModel(LogisticRegression, solver="liblinear", random_state=0)
    selector = GridSearchCVSelector(
        model=model,
        param_grid={"C": [0.1, 1.0]},
        scorer=SklearnMetric(accuracy_score),
        cv=3,
        random_state=0,
        metrics=[SklearnMetric(accuracy_score)],
    )

    selector.fit(X, y)
    best_model = selector.get_best_model()
    best_params = selector.get_best_params()
    best_metrics = selector.get_best_metrics()

    assert best_model is not None
    assert isinstance(best_params, dict) and "C" in best_params
    assert isinstance(best_metrics, dict) and "accuracy_score" in best_metrics
    yhat = best_model.predict(X)
    assert len(yhat) == len(y)

