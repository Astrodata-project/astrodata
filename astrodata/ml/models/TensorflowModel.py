import os
import random
from typing import Any, Dict, List, Optional

import keras as K
import tensorflow as tf
from tqdm import trange

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel


class TensorflowModel(BaseMlModel):
    """
    A lightweight wrapper around TensorFlow models providing a unified
    training/prediction interface and metric tracking.

    Notes
    -----
    This class expects a callable ``loss_fn`` (e.g., ``tf.keras.losses.CategoricalCrossentropy``)
    and an ``optimizer`` class (e.g., ``K.optimizers.Adam``). ``model_class`` can be
    either an instantiated ``K.Model`` or a class to be constructed from
    ``model_params``.
    """

    def __init__(
        self,
        model_class: K.Model = None,
        loss_fn: Optional[Any] = None,
        optimizer: Optional[K.optimizers.Optimizer] = None,
        model_params: Optional[Dict] = None,
        optimizer_params: Optional[Dict] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        random_state: int = random.randint(0, 2**32),
        with_weight_init: bool = False,
    ):
        super().__init__()
        self.random_state = random_state
        self.model_class = model_class
        self.model_params = model_params
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.epochs = epochs
        self.batch_size = batch_size

        self.model_ = None if not with_weight_init else self._get_model()
        self.optimizer_ = None
        self.loss_fn_ = None
        self.metrics_history_ = None
        self._val_metrics_history_ = None

    def fit(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        dataloader: Optional[tf.data.Dataset] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        metrics: Optional[List[BaseMetric]] = None,
        fine_tune: bool = False,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        # TODO
        save_every_n_epochs: Optional[int] = None,
        save_folder: Optional[str] = None,
        save_format: str = "tensorflow",
        **kwargs,
    ) -> "TensorflowModel":
        # NOTE I will be able to accept model params as keras can use also a subclass

        epochs = epochs if epochs is not None else self.epochs
        batch_size = batch_size if batch_size is not None else self.batch_size
        self.metrics_history_ = []
        self._val_metrics_history_ = (
            [] if X_val is not None and y_val is not None else None
        )

        if epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")

        if (X is None and y is None) and dataloader is None:
            raise ValueError("Either X and y or dataloader must be provided.")

        if not fine_tune or self.model_ is None:
            self.model_ = self._get_model()
            self.optimizer_ = self._get_optimizer()
            pass

        self.model_.compile(
            optimizer=self.optimizer_, loss=self.loss_fn, metrics=metrics
        )

        if dataloader is None:
            history = self.model_.fit(
                X,
                y,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(
                    (X_val, y_val) if X_val is not None and y_val is not None else None
                ),
            )
        else:
            history = self.model_.fit(
                dataloader,
                epochs=epochs,
                validation_data=(
                    (X_val, y_val) if X_val is not None and y_val is not None else None
                ),
            )

        self.metrics_history_ = history.history

        return self

    def predict(self, X, batch_size: int, **kwargs) -> Any:
        """Get class predictions (argmax of probabilities)."""
        return self._predict(X, batch_size, use_proba=False)

    def predict_proba(self, X, batch_size: int, **kwargs) -> Any:
        """Get class probabilities."""
        return self._predict(X, batch_size, use_proba=True)

    def _predict(self, X, batch_size: int, use_proba: bool) -> Any:
        """Internal prediction method that handles both probabilities and class predictions."""
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")

        # Keras predict() always returns raw model output (probabilities for classification)
        raw_predictions = self.model_.predict(X, batch_size=batch_size)

        if use_proba:
            # Return probabilities directly
            return raw_predictions
        else:
            # Return class predictions (argmax of probabilities)
            return tf.argmax(raw_predictions, axis=1).numpy()

    def score(self):
        # TODO
        pass

    def get_scorer_metric(self):
        pass

    def save(self, filepath: str, format: str = "tensorflow", **kwargs) -> None:
        # TODO
        pass

    def load(
        self, filepath: str, format: str = "tensorflow", **kwargs
    ) -> "TensorflowModel":
        # TODO
        pass

    def freeze_layers(self, layer_names: List[str]) -> None:
        # TODO
        pass

    def get_metrics(
        self,
        X,
        y,
        metrics: List[BaseMetric] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        y_pred = self.predict(X, batch_size)
        try:
            y_pred_proba = self.predict_proba(X, batch_size)
        except ValueError:
            y_pred_proba = None

        results = {}

        for metric in metrics:
            try:
                score = metric(y, y_pred_proba)
            except ValueError:
                score = metric(y, y_pred)
            results[metric.get_name()] = score
        return results

    def get_params(self, **kwargs) -> Dict[str, Any]:
        """
        Get initialization parameters for this model.

        Returns
        -------
        dict
            Parameters used to construct the model.
        """
        return {
            "model_class": self.model_class,
            "loss_fn": self.loss_fn,
            "optimizer": self.optimizer,
            "model_params": self.model_params,
            "optimizer_params": self.optimizer_params,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
        }

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """
        String representation of the wrapper and underlying class.
        """
        return f"{self.__class__.__name__}(keras_model={self.model_class.__class__.__name__})"

    def _get_model(self):
        if isinstance(self.model_class, K.Model):
            return self.model_class
        else:
            return self.model_class(**(self.model_params or {}))

    def _get_optimizer(self):
        if isinstance(self.optimizer, K.optimizers.Optimizer):
            return self.optimizer
        else:
            return self.optimizer(**(self.optimizer_params or {}))

    def clone(self) -> "TensorflowModel":
        new_instance = self.__class__(
            model_class=self.model_class,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            model_params=self.model_params,
            optimizer_params=self.optimizer_params,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )

        # Copy over any callable attributes (e.g., decorated methods)
        for attr, value in self.__dict__.items():
            if callable(value):
                setattr(new_instance, attr, value)

        return new_instance

    def get_metrics_history(self, split: str = "train") -> Dict[str, List[Any]]:
        if self.metrics_history_ is None:
            return {}

        if split == "train":
            # Return training metrics (no 'val_' prefix)
            return {
                k: v
                for k, v in self.metrics_history_.items()
                if not k.startswith("val_")
            }
        elif split == "val":
            # Return validation metrics (with 'val_' prefix, but remove prefix from keys)
            return {
                k.replace("val_", ""): v
                for k, v in self.metrics_history_.items()
                if k.startswith("val_")
            }
        else:
            raise ValueError("split must be either 'train' or 'val'")

    @property
    def has_loss_history(self) -> bool:
        return False
