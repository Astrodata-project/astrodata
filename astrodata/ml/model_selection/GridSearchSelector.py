from sklearn.model_selection import GridSearchCV
from astrodata.ml.model_selection.BaseModelSelector import BaseModelSelector
from astrodata.ml.models.BaseModel import BaseModel
from sklearn.model_selection import train_test_split
import itertools
import numpy as np

class GridSearchCVSelector(BaseModelSelector):
    def __init__(self, model:BaseModel, param_grid, scoring=None, cv=5, n_jobs=None, verbose=0, refit=True):
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
            **kwargs
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
            'param_grid': self.param_grid,
            'scoring': self.scoring,
            'cv': self.cv,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'refit': self.refit
        }
        return params


class GridSearchSelector(BaseModelSelector):
    def __init__(self, model, param_grid, scoring=None, val_size=0.2, random_state=None, metrics=None):
        super().__init__()
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.val_size = val_size
        self.random_state = random_state
        self.metrics = metrics
        self._best_model = None
        self._best_params = None
        self._best_score = None
        self._best_metrics = None

    def fit(self, X, y, *args, **kwargs):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state
        )
        param_keys = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        best_score = -np.inf
        best_params = None
        best_model = None
        best_metrics = None

        for values in itertools.product(*param_values):
            params = dict(zip(param_keys, values))
            model = self.model.clone()
            model.set_params(**params)
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
                if self.metrics:
                    best_metrics = model.get_metrics(X_val, y_val, metrics=self.metrics)

        self._best_score = best_score
        self._best_params = best_params
        self._best_model = best_model
        self._best_metrics = best_metrics
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
            "metric_classes": self.metrics,
        }
