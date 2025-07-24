import itertools

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.model_selection.BaseMlModelSelector import BaseMlModelSelector
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.tracking.Tracker import Tracker
import os


from joblib import Parallel, delayed


class GridSearchCVSelector_parallel(BaseMlModelSelector):
    """
    Performs grid search with cross-validation.
    """


    def __init__(
        self,
        model: BaseMlModel,
        param_grid: dict,
        n_jobs=os.cpu_count() - 1,   #####
        scorer: BaseMetric = None,
        cv=5,
        random_state=42,
        metrics=None,
        tracker: Tracker = None,
        log_all_models: bool = False,
    ):
        super().__init__()
        self.model = model
        self.param_grid = param_grid
        self.n_jobs = n_jobs   ######
        self.scorer = scorer
        self.cv = cv
        self.random_state = random_state
        self.metrics = metrics
        self._best_model = None
        self._best_params = None
        self._best_score = None
        self._best_metrics = None
        self.tracker = tracker
        self.log_all_models = log_all_models

    def fit(self, X, y, X_test=None, y_test=None, *args, **kwargs):
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


        ##### deve prima decidere tutte le combinazioni, così ne dà uin certo tot a ogni core
        param_combinations = [
            dict(zip(self.param_grid.keys(), values))
            for values in itertools.product(*self.param_grid.values())
        ]
        #######


        # qua valutiamo una sola combo alla volta
        def evaluate_params(params):
            fold_scores = []
            fold_metrics = []

            for train_idx, val_idx in cv_splitter.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = self.model.clone()
                model.set_params(**params)


                if self.tracker:
                    model = self.tracker.wrap_fit(
                        model,
                        X_val=X_val,
                        y_val=y_val,
                        metrics=self.metrics,
                        log_model=self.log_all_models,
                    )

                model.fit(X_train, y_train)

                if self.scorer:
                    score = model.get_metrics(X_val, y_val, metrics=[self.scorer])[self.scorer.get_name()]
                else:
                    score = model.score(X_val, y_val)
                fold_scores.append(score)

                if self.metrics:
                    fold_metrics.append(model.get_metrics(X_val, y_val, metrics=self.metrics))

            mean_score = np.mean(fold_scores)
            return params, mean_score, fold_metrics


        #qui prendo una combo alla volta, la passo a parallel che
        # le valuta dandone tot ad ogni core, poi mi memo i risultati

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_params)(params) for params in param_combinations
        )

        #cerchiamo in results la migliore combo
        for params, mean_score, fold_metrics in results:
            if (greater_is_better and mean_score > best_score) or (
                not greater_is_better and mean_score < best_score
            ):
                best_score = mean_score
                self._best_score = mean_score
                self._best_params = params
                self._best_metrics = (
                    {
                        k: sum(d[k] for d in fold_metrics) / len(fold_metrics)
                        for k in fold_metrics[0]
                    }
                    if self.metrics else None
                )

        self._best_model = self.model.clone()
        self._best_model.set_params(**self._best_params)

        if self.tracker:
            self._best_model = self.tracker.wrap_fit(
                self._best_model,
                X_test=X_test,
                y_test=y_test,
                metrics=self.metrics,
                log_model=True,
            )

        self._best_model.fit(X, y)

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
            "tracker": self.tracker,
            "log_all_models": self.log_all_models,
        }

#print(GridSearchCVSelector_parallel.__abstractmethods__)












class GridSearchSelector_parallel(BaseMlModelSelector):
    """
    Performs grid search using a single validation split.
    """

    def __init__(
        self,
        model: BaseMlModel,
        param_grid: dict,
        n_jobs: int = 1, ####
        scorer: BaseMetric = None,
        val_size=0.2,
        random_state=42,
        metrics=None,
        tracker: Tracker = None,
        log_all_models: bool = False,
    ):
        super().__init__()
        self.model = model
        self.param_grid = param_grid
        self.n_jobs = n_jobs  ######
        self.scorer = scorer
        self.val_size = val_size
        self.random_state = random_state
        self.metrics = metrics
        self.tracker = tracker
        self.log_all_models = log_all_models
        self._best_model = None
        self._best_params = None
        self._best_score = None
        self._best_metrics = None

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        *args,
        **kwargs,
    ):
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

        # Store data for refit
        self._X_train, self._y_train = X_train, y_train
        self._X_val, self._y_val = X_val, y_val


        param_combinations = [
                dict(zip(self.param_grid.keys(), values))
                for values in itertools.product(*self.param_grid.values())
            ]

        def evaluate_params(params):
            model = self.model.clone()
            model.set_params(**params)

            if self.tracker:
                model = self.tracker.wrap_fit(
                    model,
                    X_val=X_val,
                    y_val=y_val,
                    metrics=self.metrics,
                    log_model=self.log_all_models,
                )

            model.fit(X_train, y_train)

            if self.scorer:
                score = model.get_metrics(X_val, y_val, metrics=[self.scorer])[
                    self.scorer.get_name()
                ]
            else:
                score = model.score(X_val, y_val)

            metrics = (
                model.get_metrics(X_val, y_val, metrics=self.metrics)
                if self.metrics
                else None
            )

            return score, params, metrics

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_params)(params) for params in param_combinations
        )

        greater_is_better = self.scorer.greater_is_better if self.scorer else True
        best_score = -np.inf if greater_is_better else np.inf

        for score, params, metrics in results:
            if (greater_is_better and score > best_score) or (
                not greater_is_better and score < best_score
            ):
                best_score = score
                self._best_score = score
                self._best_params = params
                self._best_metrics = metrics


        # Refit best model on full data (train + val)
        X_full = np.concatenate([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])

        self._best_model = self.model.clone()
        self._best_model.set_params(**self._best_params)

        if self.tracker:
            self._best_model = self.tracker.wrap_fit(
                self._best_model,
                metrics=self.metrics,
                X_test=X_test,
                y_test=y_test,
                log_model=True,
            )

        self._best_model.fit(X_full, y_full)

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
            "tracker": self.tracker,
            "log_all_models": self.log_all_models,
        }
