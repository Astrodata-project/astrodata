import itertools
import os
import random
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone as clone_cv
from sklearn.model_selection import KFold, train_test_split

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.model_selection._utils import fit_model_score, fit_model_score_cv
from astrodata.ml.model_selection.BaseMlModelSelector import BaseMlModelSelector
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.tracking.ModelTracker import ModelTracker


def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
    if n_jobs is None:
        return 1
    if n_jobs <= 0:
        auto_jobs = (os.cpu_count() or 1) + n_jobs
        return max(1, auto_jobs)
    return n_jobs


def _param_product(param_grid: Dict[str, Iterable]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    return [
        dict(zip(keys, values)) for values in itertools.product(*param_grid.values())
    ]


def _make_cv_splitter(cv, random_state: int):
    if isinstance(cv, int):
        return KFold(n_splits=cv, shuffle=True, random_state=random_state)
    return clone_cv(cv)


class GridSearchCVSelectorParallel(BaseMlModelSelector):
    """Parallel grid search with cross-validation."""

    def __init__(
        self,
        model: BaseMlModel,
        param_grid: Dict[str, Iterable],
        n_jobs: Optional[int] = None,
        scorer: BaseMetric = None,
        cv=5,
        random_state: int = random.randint(0, 2**32),
        metrics: Optional[List[BaseMetric]] = None,
        tracker: ModelTracker = None,
        log_all_models: bool = False,
    ):
        super().__init__()
        self.model = model
        self.param_grid = param_grid
        self.n_jobs = _resolve_n_jobs(n_jobs)
        self.scorer = scorer if scorer is not None else model.get_scorer_metric()
        self.cv = cv
        self.random_state = random_state
        self.metrics = metrics if metrics is not None else []
        if self.scorer is not None and self.scorer not in self.metrics:
            self.metrics.append(self.scorer)
        self.tracker = tracker
        self.log_all_models = log_all_models
        self._best_model = None
        self._best_params = None
        self._best_score = None
        self._best_metrics = None

    def fit(self, X, y, X_test=None, y_test=None, *args, **kwargs):
        X_df = pd.DataFrame(X)
        y_df = pd.DataFrame(y)

        param_combinations = _param_product(self.param_grid)

        def evaluate_params(params: Dict[str, Any]):
            splitter = _make_cv_splitter(self.cv, self.random_state)
            current_params = dict(params)
            _, mean_metrics, mean_score = fit_model_score_cv(
                model=self.model,
                params=current_params,
                scorer=self.scorer,
                X=X_df,
                y=y_df,
                cv_splitter=splitter,
                metrics=self.metrics,
                tracker=self.tracker,
                log_models=self.log_all_models,
                tags={"stage": "training", "is_final": False, "params": current_params},
            )
            return mean_score, current_params, mean_metrics

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

        if self._best_params is None:
            raise RuntimeError(
                "Grid search failed to evaluate any parameter combinations."
            )

        if self.tracker:
            self._best_model, _, _ = fit_model_score(
                model=self.model,
                params=self._best_params,
                scorer=self.scorer,
                X_train=X_df,
                y_train=y_df,
                X_test=X_test,
                y_test=y_test,
                metrics=self.metrics,
                tracker=self.tracker,
                log_model=True,
                tags={
                    "stage": "training",
                    "is_final": True,
                    "params": self._best_params,
                },
                manual_metrics=(self._best_metrics, "val"),
            )
        else:
            self._best_model = self.model.clone()
            self._best_model.set_params(**self._best_params)
            self._best_model = self._best_model.fit(X_df, y_df)

        return self

    def get_best_model(self) -> Optional[BaseMlModel]:
        """
        Get the best model fitted on all data using the best found parameters.

        Returns
        -------
        BaseMlModel
            The best fitted model.
        """
        return self._best_model

    def get_best_params(self) -> Optional[dict]:
        """
        Get the best parameter combination found during grid search.

        Returns
        -------
        dict
            Best parameters.
        """
        return self._best_params

    def get_best_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the metrics for the best model averaged over cross-validation folds.

        Returns
        -------
        dict or None
            Averaged metrics, or None if no metrics were specified.
        """
        return self._best_metrics

    def get_params(self, **kwargs) -> Dict[str, Any]:
        """
        Get parameters of this selector instance.

        Returns
        -------
        dict
            Parameters used to initialize this object.
        """
        return {
            "model": self.model,
            "param_grid": self.param_grid,
            "scorer": self.scorer,
            "cv": self.cv,
            "random_state": self.random_state,
            "metrics": self.metrics,
            "tracker": self.tracker,
            "log_all_models": self.log_all_models,
            "n_jobs": self.n_jobs,
        }


class GridSearchSelectorParallel(BaseMlModelSelector):
    """Parallel grid search using a single validation split."""

    def __init__(
        self,
        model: BaseMlModel,
        param_grid: Dict[str, Iterable],
        n_jobs: Optional[int] = None,
        scorer: BaseMetric = None,
        val_size: Optional[float] = None,
        random_state: int = random.randint(0, 2**32),
        metrics: Optional[List[BaseMetric]] = None,
        tracker: ModelTracker = None,
        log_all_models: bool = False,
    ):
        super().__init__()
        self.model = model
        self.param_grid = param_grid
        self.n_jobs = _resolve_n_jobs(n_jobs)
        self.scorer = scorer if scorer is not None else model.get_scorer_metric()
        self.val_size = val_size
        self.random_state = random_state
        self.metrics = metrics if metrics is not None else []
        if self.scorer is not None and self.scorer not in self.metrics:
            self.metrics.append(self.scorer)
        self.tracker = tracker
        self.log_all_models = log_all_models
        self._best_model = None
        self._best_params = None
        self._best_score = None
        self._best_metrics = None

    def fit(
        self,
        X_train=None,
        y_train=None,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        dataset_train=None,
        dataset_val=None,
        dataset_test=None,
        *args,
        **kwargs,
    ) -> BaseMlModelSelector:
        """
        Run parallel grid search using a single train/validation split.

        Parameters
        ----------
        X_train : array-like, optional
            Training data features. Ignored if dataset_train is provided.
        y_train : array-like, optional
            Training data targets. Ignored if dataset_train is provided.
        X_val : array-like, optional
            Validation data features. If None and dataset_val is None, a random split is performed.
        y_val : array-like, optional
            Validation data targets. If None and dataset_val is None, a random split is performed.
        X_test : array-like, optional
            Test data features for tracking/logging (not used in selection).
        y_test : array-like, optional
            Test data targets for tracking/logging (not used in selection).
        dataset_train : Dataset, optional
            Training dataset. If provided, X_train and y_train are ignored.
        dataset_val : Dataset, optional
            Validation dataset. If provided, X_val and y_val are ignored.
        dataset_test : Dataset, optional
            Test dataset for tracking/logging (not used in selection).

        Returns
        -------
        self : object
            Fitted selector.

        Raises
        ------
        ValueError
            If neither validation data nor val_size is provided, or if neither
            (X_train, y_train) nor dataset_train is provided.
        """

        # Validate input format - either (X_train, y_train) or dataset_train
        if (X_train is None or y_train is None) and dataset_train is None:
            raise ValueError(
                "Either (X_train, y_train) or dataset_train must be provided."
            )

        # If using traditional X,y format, handle validation split
        if dataset_train is None:
            # If validation data not provided, split from training data
            if X_val is None or y_val is None:
                if self.val_size is None:
                    raise ValueError(
                        "Either val_size or validation data must be provided when using X,y format."
                    )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=self.val_size,
                    random_state=self.random_state,
                )
        else:
            # If using dataset format, validation data should also be a dataset
            if dataset_val is None:
                raise ValueError(
                    "When using dataset_train, dataset_val must also be provided."
                )

        # Store data for refit
        self._X_train, self._y_train = X_train, y_train
        self._X_val, self._y_val = X_val, y_val
        self._dataset_train = dataset_train
        self._dataset_val = dataset_val

        param_combinations = _param_product(self.param_grid)

        def evaluate_params(params: Dict[str, Any]):
            current_params = dict(params)
            _, metrics, score = fit_model_score(
                model=self.model,
                params=current_params,
                scorer=self.scorer,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                metrics=self.metrics,
                tracker=self.tracker,
                log_model=self.log_all_models,
                tags={"stage": "training", "is_final": False, "params": current_params},
                **kwargs,
            )
            return score, current_params, metrics

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

        if self._best_params is None:
            raise RuntimeError(
                "Grid search failed to evaluate any parameter combinations."
            )

        # Refit best model on full data (train + val)
        if dataset_train is not None:
            # For dataset format, we can't easily combine train and val datasets
            # So we refit on the original training dataset
            final_dataset_train = dataset_train
            final_X_train, final_y_train = None, None
            final_X_test, final_y_test = X_test, y_test
        else:
            # For X,y format, combine train and val as before
            try:
                final_X_train = pd.concat([self._X_train, self._X_val])
                final_y_train = pd.concat([self._y_train, self._y_val])
            except TypeError:
                final_X_train = np.concatenate([self._X_train, self._X_val])
                final_y_train = np.concatenate([self._y_train, self._y_val])
            final_dataset_train = None
            final_X_test, final_y_test = X_test, y_test

        if self.tracker:
            self._best_model, _, _ = fit_model_score(
                model=self.model,
                params=self._best_params,
                scorer=self.scorer,
                X_train=final_X_train,
                y_train=final_y_train,
                X_test=final_X_test,
                y_test=final_y_test,
                dataset_train=final_dataset_train,
                dataset_test=dataset_test,
                metrics=self.metrics,
                tracker=self.tracker,
                log_model=True,
                tags={
                    "stage": "training",
                    "is_final": True,
                    "params": self._best_params,
                },
                manual_metrics=(self._best_metrics, "val"),
                **kwargs,
            )
        else:
            self._best_model = self.model.clone()
            self._best_model.set_params(**self._best_params)
            if final_dataset_train is not None:
                self._best_model = self._best_model.fit(
                    dataset=final_dataset_train, **kwargs
                )
            else:
                self._best_model = self._best_model.fit(
                    final_X_train, final_y_train, **kwargs
                )

        return self

    def get_best_model(self) -> Optional[BaseMlModel]:
        """
        Get the best model fitted on all data using the best found parameters.

        Returns
        -------
        BaseMlModel
            The best fitted model.
        """
        return self._best_model

    def get_best_params(self) -> Optional[dict]:
        """
        Get the best parameter combination found during grid search.

        Returns
        -------
        dict
            Best parameters.
        """
        return self._best_params

    def get_best_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the metrics for the best model on validation data.

        Returns
        -------
        dict or None
            Metrics for the best model, or None if no metrics were specified.
        """
        return self._best_metrics

    def get_params(self, **kwargs) -> Dict[str, Any]:
        """
        Get parameters of this selector instance.

        Returns
        -------
        dict
            Parameters used to initialize this object.
        """
        return {
            "model": self.model,
            "param_grid": self.param_grid,
            "scorer": self.scorer,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "metrics": self.metrics,
            "tracker": self.tracker,
            "log_all_models": self.log_all_models,
            "n_jobs": self.n_jobs,
        }
