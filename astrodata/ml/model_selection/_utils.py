import itertools
from typing import List

import numpy as np
from sklearn.model_selection import KFold

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.tracking.ModelTracker import ModelTracker


def cross_validation_grid_search(
    model: BaseMlModel,
    param_grid: dict,
    scorer: BaseMetric,
    X,
    y,
    cv=5,
    random_state=42,
    metrics: List[BaseMetric] = None,
    tracker: ModelTracker = None,
    log_models: bool = False,
):
    greater_is_better = scorer.greater_is_better if scorer else True
    best_score = -np.inf if greater_is_better else np.inf
    best_params = None
    best_metrics = None

    # Cross-validation splitter
    if isinstance(cv, int):
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = cv

    for param_tuple in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), param_tuple))
        fold_scores = []
        fold_metrics = []

        for train_idx, val_idx in cv_splitter.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            m = model.clone()
            m.set_params(**params)

            if tracker:
                m = tracker.wrap_fit(
                    m,
                    X_val=X_val,
                    y_val=y_val,
                    metrics=metrics,
                    log_model=log_models,
                )

            m.fit(X_train, y_train)

            if scorer:
                score = m.get_metrics(X_val, y_val, metrics=[scorer])[scorer.get_name()]
            else:
                score = m.score(X_val, y_val)
            fold_scores.append(score)

            if metrics:
                fold_metrics.append(m.get_metrics(X_val, y_val, metrics=metrics))

        mean_score = np.mean(fold_scores)
        if (greater_is_better and mean_score > best_score) or (
            not greater_is_better and mean_score < best_score
        ):
            best_score = mean_score
            best_params = params
            best_metrics = (
                {
                    k: sum(d[k] for d in fold_metrics) / len(fold_metrics)
                    for k in fold_metrics[0]
                }
                if metrics
                else None
            )
    return best_score, best_params, best_metrics


def single_split_grid_search(
    model: BaseMlModel,
    param_grid: dict,
    scorer: BaseMetric,
    X_train,
    y_train,
    X_val,
    y_val,
    metrics: List[BaseMetric] = None,
    tracker: ModelTracker = None,
    log_models: bool = False,
):
    greater_is_better = scorer.greater_is_better if scorer else True
    best_score = -np.inf if greater_is_better else np.inf
    best_params = None
    best_metrics = None

    for param_tuple in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), param_tuple))
        m = model.clone()
        m.set_params(**params)

        if tracker:
            m = tracker.wrap_fit(
                m,
                X_val=X_val,
                y_val=y_val,
                metrics=metrics,
                log_model=log_models,
            )

        m.fit(X_train, y_train)

        if scorer:
            score = m.get_metrics(X_val, y_val, metrics=[scorer])[scorer.get_name()]
        else:
            score = m.score(X_val, y_val)

        if (greater_is_better and score > best_score) or (
            not greater_is_better and score < best_score
        ):
            best_score = score
            best_params = params
            best_metrics = (
                m.get_metrics(X_val, y_val, metrics=metrics) if metrics else None
            )
    return best_score, best_params, best_metrics
