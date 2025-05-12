from BaseModel import BaseModel
import tensorflow as tf
from typing import Type, Optional, Dict, Any

class TensorFlowModel(BaseModel):
    model: Optional[tf.keras.Model]
    optimizer: Optional[tf.keras.optimizers.Optimizer]
    loss: Optional[tf.keras.losses.Loss]

    def __init__(self):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.loss = None

    def initialize(
        self,
        model_class: Type[tf.keras.Model],
        optimizer_class: Type[tf.keras.optimizers.Optimizer],
        loss: tf.keras.losses.Loss,
        model_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        metrics: Optional[list] = None
    ) -> None:
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
        metrics = metrics if metrics is not None else []

        self.model = model_class(**model_kwargs)
        self.optimizer = optimizer_class(**optimizer_kwargs)
        self.loss = loss

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=metrics
        )

    def fit(self, X, y, epochs: int = 10, batch_size: int = 32, **kwargs) -> None:
        assert self.model is not None, "Call initialize() before fit()."
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, **kwargs)

    def predict(self, X, **kwargs):
        assert self.model is not None, "Call initialize() before predict()."
        return self.model.predict(X, **kwargs)

    def save(self, filepath: str, **kwargs) -> None:
        assert self.model is not None, "Call initialize() before save()."
        self.model.save(filepath, **kwargs)

    def load(self, filepath: str, **kwargs) -> None:
        self.model = tf.keras.models.load_model(filepath, **kwargs)

