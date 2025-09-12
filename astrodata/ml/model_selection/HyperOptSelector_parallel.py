import random
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, tpe
from hyperopt.pyll.base import Apply
from sklearn.model_selection import KFold, train_test_split

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.model_selection._utils import fit_model_score, fit_model_score_cv
from astrodata.ml.model_selection.BaseMlModelSelector import BaseMlModelSelector
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.tracking.ModelTracker import ModelTracker
from astrodata.utils.logger import setup_logger


from hyperopt.mongoexp import MongoTrials
from datetime import datetime
import subprocess
import platform

import time

logger = setup_logger(__name__)


class HyperOptSelector_(BaseMlModelSelector):
    """
    HyperOptSelector performs hyperparameter optimization using hyperopt.
    """

    def __init__(
        self,
        n_cores: int,

        param_space: dict,  # hyperopt search space

        mongo_url: Optional[str] = None,
        exp_key: Optional[str] = None,


        show_worker_terminal: bool = True,

        #model_mapping=None,
        model_mapping: Optional[dict] = None,
        model_list=None,
        ######################### fine cose aggiunte

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
        """
        Initialize the HyperOptSelector.

        Parameters
        ----------
        n_cores: int
            Number of cores the user chooses to use
        param_grid : dict
            Dictionary with parameter search spaces as shown in https://hyperopt.github.io/hyperopt/getting-started/search_spaces/.
        scorer : BaseMetric, optional
            The metric used to select the best model. If None, model's default score method is used.
        use_cv: bool
            Wether to use cross validation or regular validation split.
        cv : int or cross-validation splitter (default=5)
            Number of folds (int) or an object that yields train/test splits.
        max_evals: int
            Maximum number of evaluations hyperopt can run.
        random_state : int, optional
            Random seed for reproducibility.
        metrics : list of BaseMetric, optional
            Additional metrics to evaluate on validation folds.
        tracker : ModelTracker, optional
            Optional experiment/model tracker for logging.
        log_all_models : bool, optional
            If True, logs all models, not just the best one.
        """


        self.n_cores = n_cores
        self.mongo_url = mongo_url
        self.show_worker_terminal = show_worker_terminal
        self.exp_key = exp_key
        self.model_mapping = model_mapping
        self.model_list = model_list
        ########fine nuove aggiunte

        self.param_space = param_space
        #self.scorer = scorer
        self.use_cv = use_cv
        self.cv = cv
        self.val_size = val_size
        self.max_evals = max_evals
        self.random_state = random_state
        #self.metrics = metrics or []
        #if scorer is not None and scorer not in self.metrics:
        #    self.metrics.append(scorer)



        #### new scorer
        self.scorer = scorer
        self.metrics = metrics or []
        if self.scorer is not None and self.scorer not in self.metrics:
            self.metrics.append(self.scorer)
        ###########



        self.tracker = tracker
        self.log_all_models = log_all_models

        self._best_model = None
        self._best_params = None
        self._best_metrics = None




    def _objective(
        self, params: Dict[str, Any], X, y, X_val=None, y_val=None
    ) -> Dict[str, Any]:
        # hyperopt passes numpy floats, so cast where needed
        params_t = params.copy()


        """cambiare per serializzare
        model = params_t.pop("model") 
        
        if not isinstance(model, BaseMlModel):
            raise TypeError(f"{model} is not a BaseMlModel instance")
        """





        model_key = params_t.pop("model")
        model = self.model_mapping[model_key]




        #nuovo scorer
        if self.scorer is None:
            if model_key and self.model_mapping and model_key in self.model_mapping:
                model_cls = self.model_mapping[model_key]
                if hasattr(model_cls, "default_scorer") and callable(model_cls.default_scorer):
                    self.scorer = model_cls.default_scorer()
                    if self.scorer not in self.metrics:
                        self.metrics.append(self.scorer)







        if self.use_cv:

            cv_splitter = KFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )

            m, metrics, score = fit_model_score_cv(
                model,
                params_t,
                self.scorer,
                X,
                y,
                cv_splitter=cv_splitter,
                metrics=self.metrics,
                tracker=self.tracker,
                log_models=self.log_all_models,
                tags={"stage": "training", "is_final": False, "params": params},
            )
        else:
            if X_val is None or y_val is None:

                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.val_size, random_state=self.random_state
                )
            else:
                X_train, y_train = X, y

            m, metrics, score = fit_model_score(
                model,
                params_t,
                self.scorer,
                X_train,
                y_train,
                X_val,
                y_val,
                metrics=self.metrics,
                tracker=self.tracker,
                log_model=self.log_all_models,
                tags={"stage": "training", "is_final": False, "params": params},
            )

        greater_is_better = self.scorer.greater_is_better if self.scorer else True
        loss = -score if greater_is_better else score
        return {"loss": loss, "status": STATUS_OK, "metrics": metrics, "params": params}

    def fit(
        self, X, y, X_val=None, y_val=None, X_test=None, y_test=None, *args, **kwargs
    ) -> "HyperOptSelector":





        if self.n_cores == 1:
            trials = Trials()
        else:


            if self.mongo_url is None:
                raise ValueError("Mongo URL must be provided when using multiple cores.")


            #start the MongoDB service
            #TODO dà errore se non si avvia pycharm come amministratore
            os_name = platform.system()
            if os_name == "Windows":
                subprocess.run("net start MongoDB", shell=True)
            elif os_name == "Linux":
                subprocess.run(["systemctl", "start", "mongod"])
            elif os_name == "Darwin":  # macOS
                subprocess.run(["brew", "services", "start", "mongodb-community"])



            # workers url and hyperopt url are different (they need it written down in a slightly different way)
            mongo_url = self.mongo_url #complete
            mongo_url_worker = mongo_url.rsplit("/", 1)[0] #for workers

            exp_key = f"exp_linearsvc_{int(time.time())}"



            # Mongo workers on duty



            # define strategies

            def win_show():
                cmd = f'hyperopt-mongo-worker --mongo={mongo_url_worker} --exp-key={exp_key}'
                subprocess.Popen(f'start cmd /k "{cmd}"', shell=True)

            def win_hide():
                cmd = f'hyperopt-mongo-worker --mongo={mongo_url_worker} --exp-key={exp_key}'
                subprocess.Popen(["cmd", "/c", cmd], creationflags=subprocess.CREATE_NO_WINDOW)

            def unix_show():
                cmd = f'hyperopt-mongo-worker --mongo={mongo_url_worker} --exp-key={exp_key}'
                subprocess.Popen(['x-terminal-emulator', '-e', cmd])

            def unix_hide():
                cmd = f'hyperopt-mongo-worker --mongo={mongo_url_worker} --exp-key={exp_key}'
                subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


            strategies = {
                ("Windows", True): win_show,
                ("Windows", False): win_hide,
                ("Linux", True): unix_show,
                ("Linux", False): unix_hide,
                ("Darwin", True): unix_show,
                ("Darwin", False): unix_hide,
            }


            strategy = strategies.get((os_name, self.show_worker_terminal))
            if strategy is None:
                raise NotImplementedError(f"No strategy for OS={os_name}, terminal={self.show_worker_terminal}")

            # start workers
            for _ in range(self.n_cores):
                strategy()







            trials = MongoTrials(mongo_url , exp_key)



        #debug da cancellare
        print("DEBUG param_space:", self.param_space)
        assert all(isinstance(m, str) for m in self.model_list), f"model_names contiene non-stringhe: {self.model_names}"
        print("DEBUG model_names:", self.model_list)
        print("DEBUG param_space OK")



        best_params = fmin(
            fn=lambda params: self._objective(params, X, y, X_val, y_val),
            space=self.param_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            rstate=np.random.default_rng(self.random_state),
        )

        # Evaluate best to get metrics
        if self.use_cv:
            X_full, y_full = X, y
        else:
            if X_val is not None and y_val is not None:
                X_full = pd.concat([X, X_val])
                y_full = pd.concat([y, y_val])
            else:
                X_full, y_full = X, y

        # Train best model on all data
        self._best_metrics, self._best_params = _getBestMetricsParamsfromTrials(trials)





        best_params_t = self._best_params.copy()

        model_key = best_params_t.pop("model")
        model = self.model_mapping[model_key]







        if self.tracker:


            self._best_model, _, _ = fit_model_score(
                model=model,#best_params_t.pop("model"),
                params=best_params_t,
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
                    "params": self._best_params,
                },
                manual_metrics=(self._best_metrics, "val"),
            )

            """
            else:
                self._best_model = best_params_t.pop("model").clone()
                self._best_model.set_params(**best_params_t)
                self._best_model = self._best_model.fit(X_full, y_full)
            """


        else:
            self._best_model = model
            self._best_model.set_params(**best_params_t)
            self._best_model = self._best_model.fit(X_full, y_full)

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


def _getBestMetricsParamsfromTrials(trials):
    valid_trial_list = [
        trial for trial in trials if STATUS_OK == trial["result"]["status"]
    ]
    losses = [float(trial["result"]["loss"]) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj["result"]["metrics"], best_trial_obj["result"]["params"]
