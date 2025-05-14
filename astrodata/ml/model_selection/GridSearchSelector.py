from abc import ABC
from sklearn.model_selection import GridSearchCV
from astrodata.ml.model_selection.BaseModelSelector import BaseModelSelector
from astrodata.ml.models.BaseModel import BaseModel

class GridSearchSelector(BaseModelSelector):
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
