from typing import Any, Dict, Optional

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, tpe
from hyperopt.pyll.base import Apply

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.model_selection._utils import (
    cross_validation_grid_search,
    single_split_grid_search,
)
from astrodata.ml.model_selection.BaseMlModelSelector import BaseMlModelSelector
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.tracking.ModelTracker import ModelTracker


class HyperOptSelector(BaseMlModelSelector):
    """
    HyperOptSelector performs hyperparameter optimization using hyperopt.
    """

    def __init__(
        self,
        param_space: dict,  # hyperopt search space
        scorer: Optional[BaseMetric] = None,
        use_cv: bool = True,
        cv: int = 5,
        val_size: float = 0.2,
        max_evals: int = 50,
        random_state: int = 42,
        metrics: Optional[list] = None,
        tracker: Optional[ModelTracker] = None,
        log_all_models: bool = False,
    ):
        self.param_space = param_space
        self.scorer = scorer
        self.use_cv = use_cv
        self.cv = cv
        self.val_size = val_size
        self.max_evals = max_evals
        self.random_state = random_state
        self.metrics = metrics or []
        if scorer is not None and scorer not in self.metrics:
            self.metrics.append(scorer)
        self.tracker = tracker
        self.log_all_models = log_all_models

        self._best_model = None
        self._best_params = None
        self._best_metrics = None

    def _objective(
        self, params: Dict[str, Any], X, y, X_val=None, y_val=None
    ) -> Dict[str, Any]:
        # hyperopt passes numpy floats, so cast where needed
        params = {
            k: (
                int(v) if isinstance(v, np.generic) and isinstance(v.item(), int) else v
            )
            for k, v in params.items()
        }
        model = params.pop("model")

        if not isinstance(model, BaseMlModel):
            raise TypeError(f"{model} is not a BaseMlModel instance")

        if self.use_cv:
            score, _, _ = cross_validation_grid_search(
                model,
                {k: [v] for k, v in params.items()},  # wrap values in lists
                self.scorer,
                X,
                y,
                cv=self.cv,
                random_state=self.random_state,
                metrics=self.metrics,
                tracker=self.tracker,
                log_models=self.log_all_models,
            )
        else:
            if X_val is None or y_val is None:
                from sklearn.model_selection import train_test_split

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.val_size, random_state=self.random_state
                )
            else:
                X_train, y_train = X, y

            score, _, _ = single_split_grid_search(
                model,
                {k: [v] for k, v in params.items()},
                self.scorer,
                X_train,
                y_train,
                X_val,
                y_val,
                metrics=self.metrics,
                tracker=self.tracker,
                log_models=self.log_all_models,
            )

        greater_is_better = self.scorer.greater_is_better if self.scorer else True
        loss = -score if greater_is_better else score
        return {"loss": loss, "status": STATUS_OK}

    def fit(
        self, X, y, X_val=None, y_val=None, X_test=None, y_test=None, *args, **kwargs
    ) -> "HyperOptSelector":
        X = np.asarray(X)
        y = np.asarray(y)
        trials = Trials()
        best_params = fmin(
            fn=lambda params: self._objective(params, X, y, X_val, y_val),
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_state),
        )

        # Evaluate best to get metrics
        best_params = {
            k: (
                int(v) if isinstance(v, np.generic) and isinstance(v.item(), int) else v
            )
            for k, v in best_params.items()
        }
        if self.use_cv:
            X_full, y_full = X, y
        else:
            if X_val is not None and y_val is not None:
                X_full = np.concatenate([X, X_val])
                y_full = np.concatenate([y, y_val])
            else:
                X_full, y_full = X, y

        self._best_params = _map_hyperopt_best_params(best_params, self.param_space)

        # Train best model on all data
        best_model = self._best_params.pop("model")
        if hasattr(best_model, "clone"):
            self._best_model = best_model.clone()
        else:
            import copy

            self._best_model = copy.deepcopy(best_model)
        self._best_model.set_params(**self._best_params)
        if self.tracker:
            self._best_model = self.tracker.wrap_fit(
                self._best_model,
                X_test=X_test,
                y_test=y_test,
                metrics=self.metrics,
                log_model=True,
            )
        self._best_model.fit(X_full, y_full)
        self._best_metrics = self._best_model.get_metrics(
            X=X_test, y=y_test, metrics=self.metrics
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
            "param_space": self.param_space,
            "scorer": self.scorer,
            "use_cv": self.use_cv,
            "cv": self.cv,
            "val_size": self.val_size,
            "max_evals": self.max_evals,
            "random_state": self.random_state,
            "metrics": self.metrics,
            "tracker": self.tracker,
            "log_all_models": self.log_all_models,
        }


def _map_hyperopt_best_params(best_params: dict, search_space: dict) -> dict:
    """
    Maps indices in best_params back to actual values from hp.choice in search_space.
    Handles nested and scope-wrapped choices.
    """
    mapped = {}
    for k, v in best_params.items():
        space = search_space[k]
        # Unwrap "scope" functions (like scope.int)
        node = space
        while isinstance(node, Apply) and node.name.startswith("scope"):
            node = node.pos_args[0]
        # Now check if it's an hp.choice
        if isinstance(node, Apply) and node.name == "switch":
            # node.pos_args[1:] are the choices
            choices = node.pos_args[1:]
            mapped[k] = choices[v]  # v is the index
            # If choices are themselves Apply nodes (e.g., lists), evaluate them
            if hasattr(mapped[k], "eval"):
                mapped[k] = mapped[k].eval()
        else:
            mapped[k] = v
    return mapped
