import os
import random
from typing import Any, Dict, List, Optional

import keras as K
import tensorflow as tf
from tqdm import trange

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel


class KerasModel(BaseMlModel):
    """
    A lightweight wrapper around Keras models providing a unified
    training/prediction interface and metric tracking.

    Notes
    -----
    This class expects a callable ``loss_fn`` (e.g., ``K.losses.CategoricalCrossentropy``)
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
        device: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        fine_tune: bool = False,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        save_every_n_epochs: Optional[int] = None,
        save_folder: Optional[str] = None,
        save_format: str = "tensorflow",
        **kwargs,
    ) -> "KerasModel":
        
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

        if dataloader is None:
            if not isinstance(X, K.KerasTensor):
                X = K.KerasTensor(X, dtype="float32")
            if not isinstance(y, K.KerasTensor):
                y = K.KerasTensor(y, dtype="long")
            if device is None:
                device = self.device

            dataset = tf.data.Dataset((X.to(device), y.to(device)))

        if not fine_tune or self.model_ is None:
            #TODO handle when we want to finetune model rather than start from zero
            pass

        #TODO add training for keras using keras.Model.fit
        return self

    def predict(
        self, X, batch_size: int, device: Optional[str] = None, **kwargs
    ) -> Any:
        return self._predict(X, batch_size, device, use_proba=False)

    def predict_proba(
        self, X, batch_size: int, device: Optional[str] = None, **kwargs
    ) -> Any:
        return self._predict(X, batch_size, device, use_proba=True)

    def _predict(
        self, X, batch_size: int, device: Optional[str], use_proba: bool
    ) -> Any:
        #if self.model_ is None:
        #    raise RuntimeError("Model is not fitted yet.")
#
        #device = device or self.device
        #self.model_.eval()
#
        #if not isinstance(X, torch.Tensor) and not isinstance(X, DataLoader):
        #    X = torch.tensor(X, dtype=torch.float32)
#
        #dataloader = (
        #    DataLoader(X, batch_size=batch_size, shuffle=False)
        #    if not isinstance(X, DataLoader)
        #    else X
        #)
        #outputs = []
#
        #with torch.no_grad():
        #    for batch_X in dataloader:
        #        batch_X = batch_X.to(device)
        #        out = self.model_(batch_X)
        #        try:
        #            outputs.append(out.cpu())
        #        except AttributeError:
        #            outputs.append(out)
#
        #outputs = torch.cat(outputs, dim=0)
#
        #if use_proba:
        #    if outputs.shape[-1] == 1:
        #        probs = torch.sigmoid(outputs)
        #        probs = torch.cat([1 - probs, probs], dim=1)  # shape: [N, 2]
        #    else:
        #        probs = F.softmax(outputs, dim=1)
        #    return probs.numpy()
#
        #if outputs.shape[-1] == 1:
        #    return outputs.numpy().ravel()
        #return outputs.argmax(dim=1).numpy()
        
        #TODO handle basic predict that covers both predict and predict_proba
        pass

    def score(self):
        pass

    def get_scorer_metric(self):
        pass

    def save(self, filepath: str, format: str = "tensorflow", **kwargs) -> None:
        #TODO
        pass

    def load(self, filepath: str, format: str = "tensorflow", **kwargs) -> "KerasModel":
        #TODO
        pass

    def freeze_layers(self, layer_names: List[str]) -> None:
        #TODO
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

    def _get_optimizer(self, model):
        if isinstance(self.optimizer, K.Optimizer):
            return self.optimizer
        else:
            return self.optimizer(model.parameters(), **(self.optimizer_params or {}))

    def clone(self) -> "KerasModel":

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
        #TODO
        pass

    @property
    def has_loss_history(self) -> bool:
        return False
