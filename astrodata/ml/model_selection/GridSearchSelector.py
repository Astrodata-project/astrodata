import itertools

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.model_selection.BaseModelSelector import BaseModelSelector
from astrodata.ml.models.BaseModel import BaseModel


class GridSearchCVSelector(BaseModelSelector):
    """
    Performs grid search with cross-validation.
    """

    def __init__(
        self,
        model: BaseModel,
        param_grid: dict,
        scorer: BaseMetric = None,
        cv=5,
        random_state=42,
        metrics=None,
    ):
        super().__init__()
        self.model = model
        self.param_grid = param_grid
        self.scorer = scorer
        self.cv = cv
        self.random_state = random_state
        self.metrics = metrics
        self._best_model = None
        self._best_params = None
        self._best_score = None
        self._best_metrics = None

    def fit(self, X, y, *args, **kwargs):
        greater_is_better = self.scorer.greater_is_better if self.scorer else True
        best_score = -np.inf if greater_is_better else np.inf

        X = np.asarray(X)
        y = np.asarray(y)

        # Cross-validation splitter
        if isinstance(self.cv, int):
            cv_splitter = KFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
        else:
            cv_splitter = self.cv

        for param_tuple in itertools.product(*self.param_grid.values()):
            params = dict(zip(self.param_grid.keys(), param_tuple))
            fold_scores = []
            fold_metrics = []

            for train_idx, val_idx in cv_splitter.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = self.model.clone()
                model.set_params(**params)
                model.fit(X_train, y_train)

                # Score calculation
                if self.scorer:
                    score = model.get_metrics(X_val, y_val, metrics=[self.scorer])[
                        self.scorer.get_name()
                    ]
                else:
                    score = model.score(X_val, y_val)
                fold_scores.append(score)

                if self.metrics:
                    fold_metrics.append(
                        model.get_metrics(X_val, y_val, metrics=self.metrics)
                    )

            mean_score = np.mean(fold_scores)

            if (greater_is_better and mean_score > best_score) or (
                not greater_is_better and mean_score < best_score
            ):
                self._best_score = best_score = mean_score
                self._best_params = params
                # Retrain best model on all data
                self._best_model = self.model.clone()
                self._best_model.set_params(**params)
                self._best_model.fit(X, y)
                self._best_metrics = (
                    np.mean(fold_metrics, axis=0) if self.metrics else None
                )

        return self

    def get_best_model(self):
        return self._best_model

    def get_best_params(self):
        return self._best_params

    def get_best_metrics(self):
        return self._best_metrics

    def get_params(self, **kwargs):
        return {
            "model": self.model,
            "param_grid": self.param_grid,
            "scorer": self.scorer,
            "cv": self.cv,
            "random_state": self.random_state,
            "metrics": self.metrics,
        }


class GridSearchSelector(BaseModelSelector):
    """
    Performs grid search using a single validation split.
    """

    def __init__(
        self,
        model: BaseModel,
        param_grid: dict,
        scorer: BaseMetric = None,
        val_size=0.2,
        random_state=42,
        metrics=None,
    ):
        super().__init__()
        self.model = model
        self.param_grid = param_grid
        self.scorer = scorer
        self.val_size = val_size
        self.random_state = random_state
        self.metrics = metrics
        self._best_model = None
        self._best_params = None
        self._best_score = None
        self._best_metrics = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, *args, **kwargs):
        # If validation data not provided, split from training data
        if X_val is None or y_val is None:
            if self.val_size is None:
                raise ValueError("Either val_size or validation data must be provided.")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.val_size,
                random_state=self.random_state,
            )

        greater_is_better = self.scorer.greater_is_better if self.scorer else True
        best_score = -np.inf if greater_is_better else np.inf

        for param_tuple in itertools.product(*self.param_grid.values()):
            params = dict(zip(self.param_grid.keys(), param_tuple))
            model = self.model.clone()
            model.set_params(**params)
            model.fit(X_train, y_train)

            # Score calculation
            if self.scorer:
                score = model.get_metrics(X_val, y_val, metrics=[self.scorer])[
                    self.scorer.get_name()
                ]
            else:
                score = model.score(X_val, y_val)

            if (greater_is_better and score > best_score) or (
                not greater_is_better and score < best_score
            ):
                self._best_score = best_score = score
                self._best_params = params
                self._best_model = model
                self._best_metrics = (
                    model.get_metrics(X_val, y_val, metrics=self.metrics)
                    if self.metrics
                    else None
                )

        return self

    def get_best_model(self):
        return self._best_model

    def get_best_params(self):
        return self._best_params

    def get_best_metrics(self):
        return self._best_metrics

    def get_params(self, **kwargs):
        return {
            "model": self.model,
            "param_grid": self.param_grid,
            "scorer": self.scorer,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "metrics": self.metrics,
        }
