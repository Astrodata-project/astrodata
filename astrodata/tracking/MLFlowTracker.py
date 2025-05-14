import mlflow
import functools
from astrodata.tracking.BaseTracker import BaseTracker
from astrodata.ml.models.BaseModel import BaseModel
from astrodata.ml.metrics.BaseMetric import BaseMetric
from typing import List
from abc import ABC, abstractmethod
import os

class SklearnMLflowTracker(BaseTracker):
    def __init__(
        self,
        log_model=False,
        run_name=None,
        experiment_name=None,
        extra_tags=None,
        tracking_uri=None,
        tracking_username=None,
        tracking_password=None,
    ):
        super().__init__()
        self.log_model = log_model
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.extra_tags = extra_tags if extra_tags is not None else {}

        self.tracking_uri = tracking_uri
        self.tracking_username = tracking_username
        self.tracking_password = tracking_password

        self._configure_mlflow_tracking()

    def _configure_mlflow_tracking(self):
        """Set up MLflow tracking server and authentication if provided."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.tracking_username
        if self.tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.tracking_password
            
    def wrap_fit(self, model:BaseModel, input_example=None, X_test=None, y_test=None, metric_classes:List[BaseMetric]=None):
        orig_fit = model.fit

        @functools.wraps(orig_fit)
        def fit_with_tracking(X, y, *args, **kwargs):

            mlflow.set_experiment(self.experiment_name)
            with mlflow.start_run(run_name=self.run_name):
                mlflow.set_tags(self.extra_tags)
                try:
                    params = model.get_params()
                    mlflow.log_params(params)
                except Exception:
                    pass

                result = orig_fit(X, y, *args, **kwargs)
                if self.log_model:
                    try:
                        mlflow.sklearn.log_model(
                            model, 
                            "model", 
                            input_example= input_example if input_example is not None else None
                        )
                    except Exception:
                        pass
                try:
                    if hasattr(model, "get_metrics") and X_test is not None and y_test is not None:
                        metrics = model.get_metrics(X_test=X_test, y_test=y_test, metric_classes=metric_classes)
                        mlflow.log_metrics(metrics)
                except Exception:
                    pass
                return result

        model.fit = fit_with_tracking
        return model

