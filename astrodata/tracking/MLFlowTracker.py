import functools
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import mlflow

from astrodata.ml.metrics._utils import get_loss_func
from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models import BaseMlModel
from astrodata.tracking.ModelTracker import ModelTracker
from astrodata.utils.logger import setup_logger

logger = setup_logger(__name__)

warnings.filterwarnings("ignore", module=r"mlflow.*")


class MlflowBaseTracker(ModelTracker):
    """
    Base tracker class for MLflow experiment tracking.

    Handles MLflow configuration and provides base methods for registering tracked models.
    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        extra_tags: Optional[dict] = None,
        tracking_uri: Optional[str] = None,
        tracking_username: Optional[str] = None,
        tracking_password: Optional[str] = None,
    ):
        """
        Initialize MlflowBaseTracker.

        Parameters
        ----------
        run_name : str, optional
            Name for MLflow run.
        experiment_name : str, optional
            Name of the MLflow experiment.
        extra_tags : dict, optional
            Extra tags to log with the run.
        tracking_uri : str, optional
            MLflow tracking server URI.
        tracking_username : str, optional
            Username for authentication (if needed).
        tracking_password : str, optional
            Password for authentication (if needed).
        """
        super().__init__()
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.extra_tags = extra_tags if extra_tags is not None else {}
        self.tracking_uri = tracking_uri
        self.tracking_username = tracking_username
        self.tracking_password = tracking_password
        self._configure_mlflow_tracking()

    def _configure_mlflow_tracking(self):
        """
        Configure MLflow tracking URI and environment variables for authentication.
        """
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.tracking_username
        if self.tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.tracking_password

    def wrap_fit(self, obj) -> BaseMlModel:
        """
        Placeholder for tracker-specific model wrapping.

        To be implemented in subclass.
        """
        pass

    def register_best_model(
        self,
        metric: BaseMetric,
        registered_model_name: Optional[str] = None,
        split_name: str = "train",
        stage: str = "Production",
    ) -> None:
        """
        Register the best model in MLflow Model Registry based on a metric.

        Parameters
        ----------
        metric : BaseMetric
            Metric used to select the best run.
        model_artifact_path : str, optional
            Path to the model artifact in MLflow run.
        registered_model_name : str, optional
            Name for the registered model. Defaults to experiment name.
        split_name : str, optional
            Which split's metric to use ('train', 'val', or 'test').
        stage : str, optional
            Model stage to assign (e.g., 'Production', 'Staging').

        Returns
        -------
        mlflow.entities.model_registry.RegisteredModelVersion
            The result of the registration.

        Raises
        ------
        ValueError
            If the experiment or suitable run is not found.
        """

        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.experiment_name}' not found.")
        experiment_id = experiment.experiment_id

        order = "DESC" if metric.greater_is_better else "ASC"
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.is_final='True'",
            order_by=[f"metrics.{metric.get_name()}_{split_name} {order}"],
            max_results=1,
        )
        if runs_df.empty:
            raise ValueError(f"No runs found in experiment '{self.experiment_name}'.")

        best_run_id = runs_df.iloc[0].run_id

        logged_models = mlflow.search_logged_models(experiment_ids=[experiment_id])
        best_model_id = logged_models[
            logged_models["source_run_id"] == best_run_id
        ].iloc[0]["model_id"]

        model_uri = f"experiments:/{experiment_id}/{best_model_id}"

        model_name = registered_model_name or self.experiment_name

        result = mlflow.register_model(model_uri, model_name)

        if stage:
            client = mlflow.tracking.MlflowClient()
            client.set_tag(best_run_id, "stage", stage)
            client.set_registered_model_alias(
                name=model_name, alias=stage, version=result.version
            )

        logger.info(
            f"Registered model '{model_name}' version {result.version} as {stage}."
        )


class SklearnMLflowTracker(MlflowBaseTracker):
    """
    Tracker for scikit-learn models with MLflow integration.

    Provides run lifecycle, parameter logging, metric logging, and optional model logging.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize SklearnMLflowTracker.

        Parameters are passed to MlflowBaseTracker.
        """
        super().__init__(*args, **kwargs)

    def wrap_fit(
        self,
        model: BaseMlModel,
        X_test=None,
        y_test=None,
        X_val=None,
        y_val=None,
        metrics: Optional[List[BaseMetric]] = None,
        log_model: bool = False,
        tags: Dict[str, Any] = {},
        manual_metrics: Tuple[Dict[str, Any], str] = None,
    ) -> BaseMlModel:
        """
        Wrap a BaseMlModel's fit method to perform MLflow logging.

        Parameters
        ----------
        model : BaseMlModel
            The model to wrap.
        X_test : array-like, optional
            Test data for metric logging.
        y_test : array-like, optional
            Test labels for metric logging.
        X_val : array-like, optional
            Validation data for metric logging.
        y_val : array-like, optional
            Validation labels for metric logging.
        metrics : list of BaseMetric, optional
            Metrics to log. If missing, a default loss metric is added.
        log_model : bool, optional
            If True, log the fitted model as an MLflow artifact.
        tags: Dict[str, Any] default {}
            Any additional tags that should be added to the model. By default the tag "is_final" is set as equal to log_model so that
            any logged model is considered as a candidate for production (for register_best_model) unless specified otherwise
            (e.g. in the model selectors for intermediate steps)

        Returns
        -------
        BaseMlModel
            A new instance of the model with an MLflow-logging fit method.
        """
        orig_class = model.__class__
        if "is_final" not in tags.keys():
            tags["is_final"] = log_model
        tracker = self
        metrics = metrics or []

        @functools.wraps(orig_class.fit)
        def fit_with_tracking(self, X, y, *args, **kwargs):
            """
            Fit method replacement that logs parameters, metrics, and model to MLflow.

            Parameters
            ----------
            X : array-like
                Training features.
            y : array-like
                Training targets.
            tags : Dict[str, Any]
                A dictionary of tags that should be logged with the model.
            *args, **kwargs
                Additional arguments for the fit method.

            Returns
            -------
            self
                Fitted model instance.
            """
            mlflow.set_experiment(tracker.experiment_name)
            with mlflow.start_run(run_name=tracker.run_name):
                mlflow.set_tags({**tags, **tracker.extra_tags})
                try:
                    params = self.get_params()
                    mlflow.log_params(params)
                except Exception as e:
                    logger.error(f"Could not log params: {e}")

                result = orig_class.fit(self, X, y, *args, **kwargs)

                # Optionally log model
                if log_model:
                    try:
                        mlflow.sklearn.log_model(
                            self,
                            name="model",
                            signature=mlflow.models.infer_signature(
                                model_input=X[:5], model_output=y[:5]
                            ),
                        )
                    except Exception as e:
                        logger.error(f"Could not log model: {e}")

                _log_metrics_and_loss_sklearn(X, y, self, metrics, "train")
                _log_metrics_and_loss_sklearn(X_test, y_test, self, metrics, "test")
                _log_metrics_and_loss_sklearn(X_val, y_val, self, metrics, "val")

                if manual_metrics is not None:
                    _log_metrics_manual(*manual_metrics)

                return result

        # Dynamically subclass model to override fit
        class SklearnMLflowWrappedModel(orig_class):
            pass

        SklearnMLflowWrappedModel.fit = fit_with_tracking

        # Return a new instance with the same parameters as the original model
        return SklearnMLflowWrappedModel(**model.get_params())


class PytorchMLflowTracker(MlflowBaseTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def wrap_fit(
        self,
        model: BaseMlModel,
        X_test=None,
        y_test=None,
        X_val=None,
        y_val=None,
        metrics: Optional[List[BaseMetric]] = None,
        log_model: bool = False,
        tags: Dict[str, Any] = {},
        manual_metrics: Tuple[Dict[str, Any], str] = None,
    ) -> BaseMlModel:

        orig_class = model.__class__
        if "is_final" not in tags.keys():
            tags["is_final"] = log_model
        tracker = self
        metrics = metrics or []

        @functools.wraps(orig_class.fit)
        def fit_with_tracking(self, X, y, *args, **kwargs):

            mlflow.set_experiment(tracker.experiment_name)
            with mlflow.start_run(run_name=tracker.run_name):
                mlflow.set_tags({**tags, **tracker.extra_tags})
                try:
                    params = self.get_params()
                    mlflow.log_params(params)
                except Exception as e:
                    logger.error(f"Could not log params: {e}")

                result = orig_class.fit(self, X, y, metrics=metrics, *args, **kwargs)

                # Optionally log model
                if log_model:
                    try:
                        mlflow.pytorch.log_model(
                            self.model_,
                            name="model",
                            signature=mlflow.models.infer_signature(
                                model_input=X[:5], model_output=y[:5]
                            ),
                        )
                    except Exception as e:
                        logger.error(f"Could not log model: {e}")

                _log_metrics_and_loss_pytorch(X, y, self, metrics, "train")
                _log_metrics_and_loss_pytorch(X_test, y_test, self, metrics, "test")
                _log_metrics_and_loss_pytorch(X_val, y_val, self, metrics, "val")

                if manual_metrics is not None:
                    _log_metrics_manual(*manual_metrics)

                return result

        # Dynamically subclass model to override fit
        class PytorchMLflowWrappedModel(orig_class):
            pass

        PytorchMLflowWrappedModel.fit = fit_with_tracking

        # Return a new instance with the same parameters as the original model
        return PytorchMLflowWrappedModel(**model.get_params())


# Helper for metrics and loss for sklearn
def _log_metrics_and_loss_sklearn(
    X_split, y_split, model: BaseMlModel, metrics: BaseMetric, split_name: str
):
    """
    Log metrics and loss curves for a data split.

    Parameters
    ----------
    X_split : array-like
        Features.
    y_split : array-like
        Labels.
    split_name : str
        Name of the split ('train', 'val', 'test').
    """
    # Add default loss metric if not present
    loss_func = get_loss_func(model.model_)
    if loss_func is not None:
        loss_metric = SklearnMetric(loss_func)
        if loss_metric.get_name() not in [m.get_name() for m in metrics]:
            metrics.append(loss_metric)
            
    if X_split is not None and y_split is not None and hasattr(model, "get_metrics"):
        scores = model.get_metrics(X=X_split, y=y_split, metrics=metrics)
        mlflow.log_metrics({f"{k}_{split_name}": v for k, v in scores.items()})
        # Loss curve
        if model.has_loss_history:
            curves = model.get_loss_history_metrics(X_split, y_split, metrics=metrics)

            for key, value in curves.items():
                for i, loss in enumerate(value):
                    mlflow.log_metric(
                        f"{key}_{split_name}",
                        loss,
                        step=i,
                    )

# Helper for metrics and loss for sklearn
def _log_metrics_and_loss_pytorch(
    X_split, y_split, model: BaseMlModel, metrics: BaseMetric, split_name: str
):
    if X_split is not None and y_split is not None and hasattr(model, "get_metrics"):
        scores = model.get_metrics(X=X_split, y=y_split, metrics=metrics)
        mlflow.log_metrics({f"{k}_{split_name}": v for k, v in scores.items()})
        # Loss curve
        if split_name == "train" or split_name == "val":
            curves = model.get_metrics_history(split=split_name)

            for key, value in curves.items():
                for i, loss in enumerate(value):
                    mlflow.log_metric(
                        f"{key}_{split_name}",
                        loss,
                        step=i,
                    )


# Helper for manual metrics
def _log_metrics_manual(metrics: Dict[str, Any], split_name: str):
    for key, value in metrics.items():
        mlflow.log_metric(f"{key}_{split_name}", value)
