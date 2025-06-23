import itertools
from typing import List

import numpy as np
import sklearn.model_selection

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.tracking.ModelTracker import ModelTracker
from astrodata.utils.logger import setup_logger

logger = setup_logger(__name__)


def fit_model_score_cv(
    model: BaseMlModel,
    params: dict,
    scorer: BaseMetric,
    X,
    y,
    cv_splitter: sklearn.model_selection.BaseCrossValidator,
    metrics: List[BaseMetric] = None,
    tracker: ModelTracker = None,
    log_models: bool = False,
):
    fold_scores = []
    fold_metrics = []

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        m, _metrics, score = fit_model_score(
            model=model,
            params=params,
            scorer=scorer,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            metrics=metrics,
            tracker=tracker,
            log_model=log_models,
        )
        fold_scores.append(score)

        if metrics:
            fold_metrics.append(_metrics)

    mean_score = np.mean(fold_scores)
    mean_metrics = (
        {
            k: sum(d[k] for d in fold_metrics) / len(fold_metrics)
            for k in fold_metrics[0]
        }
        if metrics
        else None
    )

    return m, mean_metrics, mean_score


def fit_model_score(
    model: BaseMlModel,
    params: dict,
    scorer: BaseMetric,
    X_train,
    y_train,
    X_val,
    y_val,
    metrics: List[BaseMetric] = None,
    tracker: ModelTracker = None,
    log_model: bool = False,
):
    m = model.clone()
    m.set_params(**params)

    if tracker:
        m = tracker.wrap_fit(
            m,
            X_val=X_val,
            y_val=y_val,
            metrics=metrics,
            log_model=log_model,
        )

    m.fit(X_train, y_train)

    if scorer:
        score = m.get_metrics(X_val, y_val, metrics=[scorer])[scorer.get_name()]
    else:
        score = m.score(X_val, y_val)

    metrics = m.get_metrics(X_val, y_val, metrics=metrics) if metrics else None

    return m, metrics, score
