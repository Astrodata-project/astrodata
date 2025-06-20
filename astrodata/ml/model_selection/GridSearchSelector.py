import itertools
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.model_selection.BaseMlModelSelector import BaseMlModelSelector
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.tracking.ModelTracker import ModelTracker


class GridSearchCVSelector(BaseMlModelSelector):
    """
    GridSearchCVSelector performs exhaustive grid search over a parameter grid using cross-validation.
    """

    def __init__(
        self,
        model: BaseMlModel,
        param_grid: dict,
        scorer: BaseMetric = None,
        cv=5,
        random_state=42,
        metrics: List[BaseMetric] = None,
        tracker: ModelTracker = None,
        log_all_models: bool = False,
    ):
        """
        Initialize the GridSearchCVSelector.

        Parameters
        ----------
        model : BaseMlModel
            The model to optimize.
        param_grid : dict
            Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
        scorer : BaseMetric, optional
            The metric used to select the best model. If None, model's default score method is used.
        cv : int or cross-validation splitter (default=5)
            Number of folds (int) or an object that yields train/test splits.
        random_state : int, optional
            Random seed for reproducibility.
        metrics : list of BaseMetric, optional
            Additional metrics to evaluate on validation folds.
        tracker : ModelTracker, optional
            Optional experiment/model tracker for logging.
        log_all_models : bool, optional
            If True, logs all models, not just the best one.
        """
        self.model = model
        self.param_grid = param_grid
        self.scorer = scorer
        self.cv = cv
        self.random_state = random_state
        self.metrics = metrics or []
        if scorer is not None and scorer not in self.metrics:
            self.metrics.append(scorer)
        self._best_model = None
        self._best_params = None
        self._best_score = None
        self._best_metrics = None
        self.tracker = tracker
        self.log_all_models = log_all_models

    def fit(
        self, X, y, X_test=None, y_test=None, *args, **kwargs
    ) -> BaseMlModelSelector:
        """
        Run grid search with cross-validation.

        Parameters
        ----------
        X : array-like
            Training data features.
        y : array-like
            Training data targets.
        X_test : array-like, optional
            Test data features for tracking/logging (not used in selection).
        y_test : array-like, optional
            Test data targets for tracking/logging (not used in selection).

        Returns
        -------
        self : object
            Fitted selector.
        """
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

                if self.tracker:
                    model = self.tracker.wrap_fit(
                        model,
                        X_val=X_val,
                        y_val=y_val,
                        metrics=self.metrics,
                        log_model=self.log_all_models,
                    )

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
                self._best_metrics = (
                    {
                        k: sum(d[k] for d in fold_metrics) / len(fold_metrics)
                        for k in fold_metrics[0]
                    }
                    if self.metrics
                    else None
                )

        # Retrain best model on all data
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
        }


class GridSearchSelector(BaseMlModelSelector):
    """
    GridSearchSelector performs exhaustive grid search over a parameter grid using a single validation split.
    """

    def __init__(
        self,
        model: BaseMlModel,
        param_grid: dict,
        scorer: BaseMetric = None,
        val_size=0.2,
        random_state=42,
        metrics=None,
        tracker: ModelTracker = None,
        log_all_models: bool = False,
    ):
        """
        Initialize the GridSearchSelector.

        Parameters
        ----------
        model : BaseMlModel
            The model to optimize.
        param_grid : dict
            Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
        scorer : BaseMetric, optional
            The metric used to select the best model. If None, model's default score method is used.
        val_size : float, optional
            Fraction of training data to use as validation split (default 0.2).
        random_state : int, optional
            Random seed for reproducibility.
        metrics : list of BaseMetric, optional
            Additional metrics to evaluate on validation set.
        tracker : ModelTracker, optional
            Optional experiment/model tracker for logging.
        log_all_models : bool, optional
            If True, logs all models, not just the best one.
        """
        self.model = model
        self.param_grid = param_grid
        self.scorer = scorer
        self.val_size = val_size
        self.random_state = random_state
        self.metrics = metrics or []
        if scorer is not None and scorer not in self.metrics:
            self.metrics.append(scorer)
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
    ) -> BaseMlModelSelector:
        """
        Run grid search using a single train/validation split.

        Parameters
        ----------
        X_train : array-like
            Training data features.
        y_train : array-like
            Training data targets.
        X_val : array-like, optional
            Validation data features. If None, a random split is performed.
        y_val : array-like, optional
            Validation data targets. If None, a random split is performed.
        X_test : array-like, optional
            Test data features for tracking/logging (not used in selection).
        y_test : array-like, optional
            Test data targets for tracking/logging (not used in selection).

        Returns
        -------
        self : object
            Fitted selector.

        Raises
        ------
        ValueError
            If neither validation data nor val_size is provided.
        """
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

        for param_tuple in itertools.product(*self.param_grid.values()):
            params = dict(zip(self.param_grid.keys(), param_tuple))
            model = self.model.clone()
            model.set_params(**params)

            # Optionally wrap with tracker/logging
            if self.tracker:
                model = self.tracker.wrap_fit(
                    model,
                    X_val=X_val,
                    y_val=y_val,
                    metrics=self.metrics,
                    log_model=self.log_all_models,
                )

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
                self._best_metrics = (
                    model.get_metrics(X_val, y_val, metrics=self.metrics)
                    if self.metrics
                    else None
                )

        # Refit best model on full data (train + val)
        X_full = np.concatenate([self._X_train, self._X_val])
        y_full = np.concatenate([self._y_train, self._y_val])

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
        }
