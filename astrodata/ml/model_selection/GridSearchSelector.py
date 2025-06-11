import itertools

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.model_selection.BaseModelSelector import BaseModelSelector
from astrodata.ml.models.BaseModel import BaseModel


class GridSearchCVSelector(BaseModelSelector):
    def __init__(
        self,
        model: BaseModel,
        param_grid,
        scoring=None,
        cv=5,
        n_jobs=None,
        verbose=0,
        refit=True,
    ):
        super().__init__()
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit
        self.gs = None

    def fit(self, X, y, *args, **kwargs):
        self.gs = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            scoring=self.scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=self.refit,
            **kwargs,
        )
        self.gs.fit(X, y)

    def get_best_model(self):
        if self.gs is None:
            raise ValueError("GridSearchCV has not been fitted yet.")
        return self.gs.best_estimator_

    def get_best_params(self):
        if self.gs is None:
            raise ValueError("GridSearchCV has not been fitted yet.")
        return self.gs.best_params_

    def get_params(self, **kwargs):
        # Returns selector's config, not model's
        params = {
            "param_grid": self.param_grid,
            "scoring": self.scoring,
            "cv": self.cv,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "refit": self.refit,
        }
        return params


class GridSearchSelector(BaseModelSelector):
    def __init__(
        self,
        model,
        param_grid,
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
        self._best_model = self._best_params = self._best_score = self._best_metrics = (
            None
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, *args, **kwargs):
        if X_val is None or y_val is None:
            if self.val_size is None:
                raise ValueError("Either val_size or validation data must be provided.")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.val_size,
                random_state=self.random_state,
            )

        best_score = -np.inf if self.scorer.greater_is_better else np.inf

        for param_tuple in itertools.product(*self.param_grid.values()):
            params = dict(zip(self.param_grid.keys(), param_tuple))
            model = self.model.clone()
            model.set_params(**params)
            model.fit(X_train, y_train)
            score = (
                model.get_metrics(X_val, y_val, metrics=[self.scorer])[
                    self.scorer.get_name()
                ]
                if self.scorer
                else model.score(X_val, y_val)
            )

            if (
                score > best_score
                if self.scorer.greater_is_better
                else score < best_score
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
            "scoring": self.scoring,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "metrics": self.metrics,
        }
