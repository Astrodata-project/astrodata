import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from astrodata.ml.model_selection.BaseModelSelector import BaseModelSelector

# TODO


class HyperOptSelector(BaseModelSelector):

    def __init__(self, model, cv=5, scorer=None, metrics=None, random_state=42):

        super().__init__()
        self.model = model
        self.cv = cv
        self.scorer = scorer
        self.metrics = metrics
        self.random_state = random_state

        # inizializziamo cose
        self._best_params = None
        self._best_model = None
        self._best_metrics = None

    def fit(self, X, y, X_test=None, y_test=None, space=None, *args, **kwargs):

        # definisco SOLO come fare cross validation
        if isinstance(self.cv, int):
            cv_splitter = StratifiedKFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            cv_splitter = self.cv

        def objective(params):
            """
            #forse qua va mlflow? l'ho solo copia incollato

            with mlflow.start_run(nested=True):  # nested=True se dentro un run principale
            model = self.model.clone()
            model.set_params(**params)
            """

            model = self.model.clone()  # non alterare l'originale

            # converti parametri che vengono dallo space
            if "max_iter" in params:
                params["max_iter"] = int(params["max_iter"])

            model.set_params(**params)

            # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

            # questo valuta il modello todo: cambiare le metriche?
            scores = cross_val_score(
                model,
                self.X,
                self.y,
                cv=cv_splitter,
                scoring=make_scorer(accuracy_score),
            )

            loss = 1 - scores.mean()

            return loss

        trials = Trials()  # registra cosa succede nell'ottimizzazione, debbugging

        best_params = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,  # todo non ricordo algo cosa...
            max_evals=50,
            trials=trials,
        )

        self._best_params = best_params
        self._best_model = self.model.set_params(**best_params).fit(X, y)

        # qua salvo le metriche, però todo dobbiamo decidere quali
        if self.metrics:
            self._best_metrics = self._best_model.get_metrics(
                X, y, metrics=self.metrics
            )
        else:
            self._best_metrics = None

        """
        mlflow.log_params(params)
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("mean_accuracy", scores.mean())
        """

        return self

    def get_best_params(self):
        return self._best_params

    def get_best_model(self):
        return self._best_model

    def get_params(self, **kwargs) -> dict:
        return {
            "cv": self.cv,
            "scorer": self.scorer,
            "metrics": self.metrics,
            "random_state": self.random_state,
            "model": self.model,
        }
