import itertools
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
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
    tags: Dict[str, Any] = None,
):
    fold_scores = []
    fold_metrics = []

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

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
            tags=tags,
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
    X_train: pd.DataFrame = None,
    y_train: pd.DataFrame = None,
    X_val: pd.DataFrame = None,
    y_val: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    y_test: pd.DataFrame = None,
    dataset_train=None,
    dataset_val=None,
    dataset_test=None,
    metrics: List[BaseMetric] = None,
    tracker: ModelTracker = None,
    log_model: bool = False,
    tags: Dict[str, Any] = None,
    manual_metrics: Tuple[Dict[str, Any], str] = None,
    **kwargs,
):
    metrics_res, score = None, None
    m = model.clone()
    m.set_params(**params)

    # Handle input format - either (X_train, y_train) or dataset_train
    if dataset_train is not None:
        # For dataset input, we don't need to modify y_train
        y_train_mod = None
        y_val_mod = None
    else:
        # For (X, y) input, handle DataFrame format
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train_mod = y_train.iloc[:, 0]
        else:
            y_train_mod = y_train

        if isinstance(y_val, pd.DataFrame) and y_val.shape[1] == 1:
            y_val_mod = y_val.iloc[:, 0]
        else:
            y_val_mod = y_val

    if tracker:
        m = tracker.wrap_fit(
            m,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            metrics=metrics,
            log_model=log_model,
            tags=tags,
            manual_metrics=manual_metrics,
        )

    # Fit the model with appropriate input format
    if dataset_train is not None:
        m.fit(dataset=dataset_train, dataset_val=dataset_val, **kwargs)
    else:
        m.fit(X_train, y_train_mod, X_val=X_val, y_val=y_val_mod, **kwargs)

    if dataset_val is not None:
        if scorer:
            score = m.get_metrics(dataset=dataset_val, metrics=[scorer])[
                scorer.get_name()
            ]
        else:
            score = m.score(dataset=dataset_val)

        metrics_res = (
            m.get_metrics(dataset=dataset_val, metrics=metrics) if metrics else None
        )

        return m, metrics_res, score

    if X_val is not None and y_val is not None:
        if scorer:
            score = m.get_metrics(X=X_val, y=y_val_mod, metrics=[scorer])[
                scorer.get_name()
            ]
        else:
            score = m.score(X=X_val, y=y_val_mod)

        metrics_res = (
            m.get_metrics(X=X_val, y=y_val_mod, metrics=metrics) if metrics else None
        )

    return m, metrics_res, score
