import functools
import os
from typing import List

import mlflow
from sklearn import metrics as sklearn_metrics

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models import BaseModel
from astrodata.tracking.BaseTracker import BaseTracker
from astrodata.utils.MetricsUtils import get_loss_func


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
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.tracking_username
        if self.tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.tracking_password

    def wrap_fit(
        self,
        model: BaseModel,
        input_example=None,
        X_test=None,
        y_test=None,
        X_val=None,
        y_val=None,
        metrics: List = [],
    ):
        orig_class = model.__class__
        tracker = self

        @functools.wraps(orig_class.fit)
        def fit_with_tracking(self, X, y, *args, **kwargs):
            import mlflow

            mlflow.set_experiment(tracker.experiment_name)
            with mlflow.start_run(run_name=tracker.run_name):
                mlflow.set_tags(tracker.extra_tags)
                try:
                    params = self.get_params()
                    mlflow.log_params(params)
                except Exception:
                    pass

                result = orig_class.fit(self, X, y, *args, **kwargs)

                if tracker.log_model:
                    try:
                        mlflow.sklearn.log_model(
                            self, "model", input_example=input_example
                        )
                    except Exception:
                        pass

                # Helper to log metrics and loss curves for a data split
                def log_metrics_and_loss(X_split, y_split, split_name):
                    if (
                        X_split is not None
                        and y_split is not None
                        and hasattr(self, "get_metrics")
                    ):
                        metrics_scores = self.get_metrics(
                            X=X_split, y=y_split, metrics=metrics
                        )
                        # Prefix all metric names with the split name
                        metrics_scores = {
                            f"{name}_{split_name}": val
                            for name, val in metrics_scores.items()
                        }
                        mlflow.log_metrics(metrics_scores)

                        # Log loss curve for each metric if possible
                        for metric in metrics or []:
                            if hasattr(self, "get_loss_history"):
                                loss_curve = self.get_loss_history_metric(
                                    X_split, y_split, metric=metric
                                )
                                for i, loss in enumerate(loss_curve):
                                    mlflow.log_metric(
                                        f"{metric.get_name()}_{split_name}_step",
                                        loss,
                                        step=i,
                                    )

                loss_func = get_loss_func(self.model_)
                metrics.append(SklearnMetric(loss_func))

                log_metrics_and_loss(X, y, "train")
                log_metrics_and_loss(X_test, y_test, "test")
                log_metrics_and_loss(X_val, y_val, "val")

                return result

        # Dynamically subclass
        class SklearnMLflowWrappedModel(orig_class):
            pass

        SklearnMLflowWrappedModel.fit = fit_with_tracking

        # Return instance with same params
        return SklearnMLflowWrappedModel(**model.get_params())
