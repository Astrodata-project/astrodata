import functools
import os
from typing import List, Optional

import mlflow

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models import BaseModel
from astrodata.tracking.BaseTracker import BaseTracker
from astrodata.utils.MetricsUtils import get_loss_func


class MlflowBaseTracker(BaseTracker):
    def __init__(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        extra_tags: Optional[dict] = None,
        tracking_uri: Optional[str] = None,
        tracking_username: Optional[str] = None,
        tracking_password: Optional[str] = None,
    ):
        super().__init__()
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

    def wrap_fit(self, obj):
        pass  # To be implemented in subclass

    def register_best_model(
        self,
        metric: BaseMetric,
        model_artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        split_name: str = "train",
        stage: str = "Production",
    ):
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found.")
        experiment_id = experiment.experiment_id

        order = "DESC" if metric.greater_is_better else "ASC"
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric.get_name()}_{split_name} {order}"],
            max_results=1,
        )
        if runs_df.empty:
            raise ValueError(f"No runs found in experiment '{self.experiment_name}'.")

        best_run_id = runs_df.iloc[0].run_id
        model_name = registered_model_name or self.experiment_name
        model_uri = f"runs:/{best_run_id}/{model_artifact_path}"
        result = mlflow.register_model(model_uri, model_name)

        if stage:
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(
                name=model_name, alias=stage, version=result.version
            )
        print(f"Registered model '{model_name}' version {result.version} as {stage}.")
        return result


class SklearnMLflowTracker(MlflowBaseTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def wrap_fit(
        self,
        model: BaseModel,
        X_test=None,
        y_test=None,
        X_val=None,
        y_val=None,
        metrics: Optional[List[BaseMetric]] = None,
        log_model: bool = False,
    ):
        orig_class = model.__class__
        tracker = self
        metrics = metrics or []

        @functools.wraps(orig_class.fit)
        def fit_with_tracking(self, X, y, *args, **kwargs):
            mlflow.set_experiment(tracker.experiment_name)
            with mlflow.start_run(run_name=tracker.run_name):
                mlflow.set_tags(tracker.extra_tags)
                try:
                    params = self.get_params()
                    mlflow.log_params(params)
                except Exception as e:
                    print(f"Could not log params: {e}")

                result = orig_class.fit(self, X, y, *args, **kwargs)

                # Optionally log model
                if log_model:
                    try:
                        mlflow.sklearn.log_model(self, "model", input_example=X[:5])
                    except Exception as e:
                        print(f"Could not log model: {e}")

                # Helper for metrics and loss
                def log_metrics_and_loss(X_split, y_split, split_name):
                    if (
                        X_split is not None
                        and y_split is not None
                        and hasattr(self, "get_metrics")
                    ):
                        scores = self.get_metrics(X=X_split, y=y_split, metrics=metrics)
                        mlflow.log_metrics(
                            {f"{k}_{split_name}": v for k, v in scores.items()}
                        )
                        # Loss curve
                        for metric in metrics:
                            if hasattr(self, "get_loss_history_metric"):
                                curve = self.get_loss_history_metric(
                                    X_split, y_split, metric=metric
                                )
                                for i, loss in enumerate(curve):
                                    mlflow.log_metric(
                                        f"{metric.get_name()}_{split_name}_step",
                                        loss,
                                        step=i,
                                    )

                # Add default loss metric if not present
                loss_metric = SklearnMetric(get_loss_func(self.model_))
                if loss_metric.get_name() not in [m.get_name() for m in metrics]:
                    metrics.append(loss_metric)

                log_metrics_and_loss(X, y, "train")
                log_metrics_and_loss(X_test, y_test, "test")
                log_metrics_and_loss(X_val, y_val, "val")

                return result

        # Dynamically subclass model to override fit
        class SklearnMLflowWrappedModel(orig_class):
            pass

        SklearnMLflowWrappedModel.fit = fit_with_tracking

        # Return a new instance with the same parameters as the original model
        return SklearnMLflowWrappedModel(**model.get_params())
