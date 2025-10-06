import platform
import random
import subprocess
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, tpe
from hyperopt.mongoexp import MongoTrials
from sklearn.model_selection import KFold, train_test_split

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.model_selection._utils import fit_model_score, fit_model_score_cv
from astrodata.ml.model_selection.BaseMlModelSelector import BaseMlModelSelector
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.tracking.ModelTracker import ModelTracker
from astrodata.utils.logger import setup_logger

logger = setup_logger(__name__)


def _get_best_metrics_and_params(trials: Trials):
    valid_trial_list = [
        trial for trial in trials if STATUS_OK == trial["result"]["status"]
    ]
    if not valid_trial_list:
        raise RuntimeError("Hyperopt trials did not produce any successful evaluations.")

    losses = [float(trial["result"]["loss"]) for trial in valid_trial_list]
    best_index = int(np.argmin(losses))
    best_trial_obj = valid_trial_list[best_index]
    return best_trial_obj["result"]["metrics"], best_trial_obj["result"]["params"]


class HyperOptSelectorParallel(BaseMlModelSelector):
    """Hyperparameter optimisation with Hyperopt supporting parallel execution."""

    def __init__(
        self,
        param_space: dict,
        n_cores: int = 1,
        mongo_url: Optional[str] = None,
        exp_key: Optional[str] = None,
        show_worker_terminal: bool = True,
        model_mapping: Optional[Dict[str, BaseMlModel]] = None,
        scorer: Optional[BaseMetric] = None,
        use_cv: bool = False,
        cv: Optional[int] = 2,
        val_size: Optional[float] = 0.2,
        max_evals: int = 20,
        random_state: int = random.randint(0, 2**32),
        metrics: Optional[list] = None,
        tracker: Optional[ModelTracker] = None,
        log_all_models: bool = False,
    ):
        super().__init__()
        self.param_space = param_space
        self.n_cores = n_cores
        self.mongo_url = mongo_url
        self.exp_key = exp_key
        self.show_worker_terminal = show_worker_terminal
        self.model_mapping = model_mapping or {}
        self.scorer = scorer
        self.use_cv = use_cv
        self.cv = cv
        self.val_size = val_size
        self.max_evals = max_evals
        self.random_state = random_state
        self.metrics = metrics or []
        if self.scorer is not None and self.scorer not in self.metrics:
            self.metrics.append(self.scorer)
        self.tracker = tracker
        self.log_all_models = log_all_models

        self._best_model = None
        self._best_params = None
        self._best_metrics = None
        self._active_exp_key = None

    def _resolve_model(self, model_choice: Any) -> BaseMlModel:
        if isinstance(model_choice, BaseMlModel):
            if self.n_cores > 1:
                raise TypeError(
                    "When n_cores > 1, hyperopt param_space must encode models "
                    "using serialisable identifiers mapped through model_mapping."
                )
            return model_choice
        if isinstance(model_choice, str):
            if model_choice not in self.model_mapping:
                raise KeyError(
                    f"Model identifier '{model_choice}' not found in model_mapping."
                )
            return self.model_mapping[model_choice]
        raise TypeError(f"Unsupported model specification: {model_choice!r}")

    def _build_cv_splitter(self) -> KFold:
        if isinstance(self.cv, int):
            return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return self.cv

    def _objective(self, params: Dict[str, Any], X, y, X_val=None, y_val=None) -> Dict[str, Any]:
        params_t = dict(params)
        model_choice = params_t.pop("model")
        model = self._resolve_model(model_choice)

        logged_params = dict(params_t)
        logged_params["model"] = model_choice

        if self.use_cv:
            splitter = self._build_cv_splitter()
            _, metrics, score = fit_model_score_cv(
                model=model,
                params=params_t,
                scorer=self.scorer,
                X=X,
                y=y,
                cv_splitter=splitter,
                metrics=self.metrics,
                tracker=self.tracker,
                log_models=self.log_all_models,
                tags={"stage": "training", "is_final": False, "params": logged_params},
            )
        else:
            if X_val is None or y_val is None:
                X_train, X_val_tmp, y_train, y_val_tmp = train_test_split(
                    X, y, test_size=self.val_size, random_state=self.random_state
                )
            else:
                X_train, X_val_tmp, y_train, y_val_tmp = X, X_val, y, y_val

            _, metrics, score = fit_model_score(
                model=model,
                params=params_t,
                scorer=self.scorer,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val_tmp,
                y_val=y_val_tmp,
                metrics=self.metrics,
                tracker=self.tracker,
                log_model=self.log_all_models,
                tags={"stage": "training", "is_final": False, "params": logged_params},
            )

        greater_is_better = self.scorer.greater_is_better if self.scorer else True
        loss = -score if greater_is_better else score
        return {"loss": loss, "status": STATUS_OK, "metrics": metrics, "params": logged_params}

    def _ensure_mongo_ready(self):
        os_name = platform.system()
        try:
            if os_name == "Windows":
                subprocess.run("net start MongoDB", shell=True, check=False)
            elif os_name == "Linux":
                subprocess.run(["systemctl", "start", "mongod"], check=False)
            elif os_name == "Darwin":
                subprocess.run(["brew", "services", "start", "mongodb-community"], check=False)
        except Exception as exc:
            logger.warning("Unable to ensure MongoDB service is running: %s", exc)

    def _launch_workers(self, mongo_url_worker: str, exp_key: str):
        worker_cmd = [
            "hyperopt-mongo-worker",
            f"--mongo={mongo_url_worker}",
            f"--exp-key={exp_key}",
        ]
        for _ in range(self.n_cores):
            try:
                if platform.system() == "Windows":
                    creationflags = 0
                    if not self.show_worker_terminal and hasattr(subprocess, "CREATE_NO_WINDOW"):
                        creationflags = subprocess.CREATE_NO_WINDOW
                    subprocess.Popen(
                        ["cmd", "/c", " ".join(worker_cmd)],
                        creationflags=creationflags,
                    )
                else:
                    stdout = None
                    stderr = None
                    if not self.show_worker_terminal:
                        stdout = subprocess.DEVNULL
                        stderr = subprocess.DEVNULL
                    subprocess.Popen(worker_cmd, stdout=stdout, stderr=stderr)
            except Exception as exc:
                logger.warning("Failed to start hyperopt worker: %s", exc)

    def _prepare_trials(self) -> Trials:
        if self.n_cores <= 1:
            return Trials()
        if not self.mongo_url:
            raise ValueError("mongo_url must be provided when n_cores > 1.")

        self._ensure_mongo_ready()
        exp_key = self.exp_key or f"hyperopt_exp_{int(time.time())}"
        self._active_exp_key = exp_key
        mongo_url_worker = self.mongo_url.rsplit("/", 1)[0]
        self._launch_workers(mongo_url_worker, exp_key)
        return MongoTrials(self.mongo_url, exp_key)

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        *args,
        **kwargs,
    ) -> "HyperOptSelectorParallel":
        trials = self._prepare_trials()

        fmin(
            fn=lambda params: self._objective(params, X, y, X_val, y_val),
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_state),
        )

        if self.use_cv:
            X_full, y_full = X, y
        else:
            if X_val is not None and y_val is not None:
                try:
                    X_full = pd.concat([X, X_val])
                    y_full = pd.concat([y, y_val])
                except TypeError:
                    X_full = np.concatenate([np.asarray(X), np.asarray(X_val)])
                    y_full = np.concatenate([np.asarray(y), np.asarray(y_val)])
            else:
                X_full, y_full = X, y

        self._best_metrics, best_params_raw = _get_best_metrics_and_params(trials)

        model_choice = best_params_raw.get("model")
        model = self._resolve_model(model_choice)

        fit_params = dict(best_params_raw)
        fit_params.pop("model", None)

        if self.tracker:
            self._best_model, _, _ = fit_model_score(
                model=model,
                params=fit_params,
                scorer=self.scorer,
                X_train=X_full,
                y_train=y_full,
                X_test=X_test,
                y_test=y_test,
                metrics=self.metrics,
                tracker=self.tracker,
                log_model=True,
                tags={
                    "stage": "training",
                    "is_final": True,
                    "params": best_params_raw,
                },
                manual_metrics=(self._best_metrics, "val"),
            )
        else:
            self._best_model = model.clone()
            self._best_model.set_params(**fit_params)
            self._best_model = self._best_model.fit(X_full, y_full)

        best_params_for_user = dict(best_params_raw)
        best_params_for_user["model"] = model.clone()
        self._best_params = best_params_for_user

        return self

    def get_best_model(self) -> Optional[BaseMlModel]:
        return self._best_model

    def get_best_params(self) -> Optional[dict]:
        return self._best_params

    def get_best_metrics(self) -> Optional[Dict[str, Any]]:
        return self._best_metrics

    def get_params(self, **kwargs) -> Dict[str, Any]:
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
            "n_cores": self.n_cores,
            "mongo_url": self.mongo_url,
            "exp_key": self._active_exp_key or self.exp_key,
        }