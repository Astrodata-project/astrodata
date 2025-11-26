import os
import random
from typing import Any, Callable, Dict, List, Optional, Type, Union

import keras as K
import tensorflow as tf

from astrodata.ml.metrics import BaseMetric, TensorflowMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel


class TensorflowModel(BaseMlModel):
    """
    A lightweight wrapper around TensorFlow/Keras models providing a unified
    training/prediction interface and metric tracking.

    Notes
    -----
    This class expects a callable ``loss_fn`` (e.g., ``keras.losses.SparseCategoricalCrossentropy``)
    and an ``optimizer`` class (e.g., ``keras.optimizers.Adam``). ``model_class`` can be
    either an instantiated ``keras.Model`` or a class to be constructed from
    ``model_params``.
    """

    def __init__(
        self,
        model_class: Union[K.Model, Type[K.Model], Callable] = None,
        loss_fn: Optional[K.losses.Loss] = None,
        optimizer: Optional[K.optimizers.Optimizer] = None,
        model_params: Optional[Dict] = None,
        optimizer_params: Optional[Dict] = None,
        device: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        random_state: int = random.randint(0, 2**32),
        with_weight_init: bool = False,
    ):
        super().__init__()

        # Validate parameters
        if loss_fn is not None and not callable(loss_fn):
            raise ValueError("loss_fn must be callable")
        if optimizer is not None and not callable(optimizer):
            raise ValueError("optimizer must be callable")
        if epochs is not None and epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.random_state = random_state
        self.model_class = model_class
        self.model_params = model_params
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.with_weight_init = with_weight_init

        self.model_ = None if not with_weight_init else self._get_model()
        self.optimizer_ = None
        self.loss_fn_ = None
        self.metrics_history_ = None
        self._val_metrics_history_ = None

    def fit(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        dataset: Optional[tf.data.Dataset] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        metrics: Optional[List[TensorflowMetric]] = None,
        fine_tune: bool = False,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        dataset_val: Optional[tf.data.Dataset] = None,
        save_every_n_epochs: Optional[int] = None,
        save_folder: Optional[str] = None,
        save_format: str = "tensorflow",
        shuffle: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "TensorflowModel":
        """
        Fit the model using the provided data or dataset.

        Parameters
        ----------
        X : array-like or tf.Tensor, optional
            Training features. Ignored if ``dataset`` is provided.
        y : array-like or tf.Tensor, optional
            Training labels. Ignored if ``dataset`` is provided.
        dataset : tf.data.Dataset, optional
            Pre-built training dataset yielding ``(inputs, labels)``.
        epochs : int, optional
            Number of training epochs. Defaults to the instance value.
        batch_size : int, optional
            Batch size for training/prediction. Defaults to the instance value.
        device : str, optional
            Device to use (currently not used in TensorFlow implementation).
        metrics : list of BaseMetric, optional
            Metrics to compute during training and validation.
        fine_tune : bool, default False
            If True, reuse existing model weights/optimizer state when available.
        X_val : array-like or tf.Tensor, optional
            Validation features for epoch-wise metric tracking.
        y_val : array-like or tf.Tensor, optional
            Validation labels for epoch-wise metric tracking.
        dataset_val : tf.data.Dataset, optional
            Pre-built validation dataset yielding ``(inputs, labels)``.
        save_every_n_epochs : int, optional
            Save model every n epochs if provided.
        save_folder : str, optional
            Directory path where checkpoints will be saved.
        save_format : {"tensorflow", "h5", "savedmodel"}, default "tensorflow"
            Serialization format for checkpoints.
        shuffle : bool, default True
            Whether to shuffle the training data.
        seed : int, optional
            Random seed for reproducibility. Defaults to instance random_state.
        **kwargs
            Additional arguments passed to model.fit().

        Returns
        -------
        TensorflowModel
            The fitted model instance.
        """

        K.utils.set_random_seed(seed if seed is not None else self.random_state)

        epochs = epochs if epochs is not None else self.epochs
        self.loss_fn_ = self.loss_fn()

        batch_size = batch_size if batch_size is not None else self.batch_size
        self.metrics_history_ = []
        self._val_metrics_history_ = (
            []
            if (X_val is not None and y_val is not None) or dataset_val is not None
            else None
        )

        if epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")

        if (X is None and y is None) and dataset is None:
            raise ValueError("Either X and y or dataset must be provided.")

        if dataset is None:
            dataset = self._create_dataset(X, y)

        if dataset_val is None and X_val is not None and y_val is not None:
            dataset_val = self._create_dataset(X_val, y_val)

        if not fine_tune or self.model_ is None:
            self.model_ = self._get_model()

        if not fine_tune or self.optimizer_ is None:
            self.optimizer_ = self._get_optimizer()

        self.model_.compile(
            optimizer=self.optimizer_, loss=self.loss_fn_, metrics=metrics
        )

        # Prepare callbacks
        callbacks = kwargs.get("callbacks", [])

        # Add periodic saving callback if requested
        if save_every_n_epochs is not None and save_every_n_epochs > 0:
            if save_folder is None:
                raise ValueError(
                    "save_folder must be provided when save_every_n_epochs is specified."
                )

            # Ensure save folder exists
            os.makedirs(save_folder, exist_ok=True)

            periodic_save_callback = _PeriodicSaveCallback(
                save_every_n_epochs, save_folder, save_format
            )
            callbacks.append(periodic_save_callback)

        # Update kwargs with callbacks
        kwargs["callbacks"] = callbacks

        history = self.model_.fit(
            dataset.batch(batch_size),
            epochs=epochs,
            validation_data=(
                dataset_val.batch(batch_size) if dataset_val is not None else None
            ),
            shuffle=shuffle,
            **kwargs,
        )

        self.metrics_history_ = history.history

        return self

    def predict(
        self, data, batch_size=32, device: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Get class predictions (argmax of probabilities).

        Parameters
        ----------
        data : array-like, tf.Tensor, or tf.data.Dataset
            Features to predict. If a Dataset is provided, it should yield
            feature tensors only.
        batch_size : int, default 32
            Batch size for prediction.
        device : str, optional
            Device to use for inference (currently not used in TensorFlow implementation).
        **kwargs
            Additional arguments passed to model.predict().

        Returns
        -------
        numpy.ndarray
            Predicted class labels for classification.

        Raises
        ------
        ValueError
            If the model is not fitted yet.
        """
        return self._predict(data, batch_size, use_proba=False)

    def predict_proba(
        self, data=None, batch_size=32, device: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Get class probabilities.

        Parameters
        ----------
        data : array-like, tf.Tensor, or tf.data.Dataset, optional
            Features to predict. If a Dataset is provided, it should yield
            feature tensors only.
        batch_size : int, default 32
            Batch size for prediction.
        device : str, optional
            Device to use for inference (currently not used in TensorFlow implementation).
        **kwargs
            Additional arguments passed to model.predict().

        Returns
        -------
        numpy.ndarray
            Predicted class probabilities with shape ``[N, n_classes]``.

        Raises
        ------
        ValueError
            If the model is not fitted yet.
        """
        return self._predict(data, batch_size, use_proba=True)

    def _predict(
        self, X, batch_size: int, use_proba: bool, device: Optional[str] = None
    ) -> Any:
        """
        Internal prediction method that handles both probabilities and class predictions.

        Parameters
        ----------
        X : array-like or tf.Tensor
            Input features.
        batch_size : int
            Batch size for prediction.
        use_proba : bool
            If True, return probabilities; if False, return class predictions.
        device : str, optional
            Device to use for inference (currently not used in TensorFlow implementation).

        Returns
        -------
        numpy.ndarray
            Either class probabilities or class predictions based on use_proba.

        Raises
        ------
        ValueError
            If the model is not fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")

        raw_predictions = self.model_.predict(
            X.batch(batch_size) if isinstance(X, tf.data.Dataset) else X,
            batch_size=batch_size,
        )

        if use_proba:
            # Return probabilities directly
            return raw_predictions
        else:
            # Return class predictions (argmax of probabilities)
            return tf.argmax(raw_predictions, axis=1).numpy()

    def score(
        self,
        dataset=None,
        X=None,
        y=None,
        batch_size=None,
        device: Optional[str] = None,
        **kwargs,
    ) -> float:
        """
        Compute the average loss on the given data.

        Parameters
        ----------
        dataset : tf.data.Dataset, optional
            Pre-built dataset yielding ``(inputs, labels)``.
        X : array-like or tf.Tensor, optional
            Features. Ignored if ``dataset`` is provided.
        y : array-like or tf.Tensor, optional
            Labels. Ignored if ``dataset`` is provided.
        batch_size : int, optional
            Batch size when ``X`` and ``y`` are provided.
        device : str, optional
            Device to use for inference (currently not used in TensorFlow implementation).
        **kwargs
            Additional arguments.

        Returns
        -------
        float
            Average loss over the dataset.

        Raises
        ------
        ValueError
            If neither dataset nor (X, y) are provided.
        """
        batch_size = batch_size or self.batch_size

        if dataset is None:
            if X is None or y is None:
                raise ValueError("Either dataset or both X and y must be provided")
            # Convert X, y to dataset
            dataset = self._create_dataset(X, y)

        # Batch the dataset if not already batched
        dataset_batched = dataset.batch(batch_size)

        loss = self.model_.evaluate(dataset_batched, verbose=0)
        return loss[0] if isinstance(loss, list) else loss

    def get_scorer_metric(self):
        return self.loss_fn

    def save(self, filepath: str, format: str = "tensorflow", **kwargs) -> None:
        """
        Save the model parameters and optimizer state.

        Parameters
        ----------
        filepath : str
            Destination path.
        format : {"tensorflow", "h5", "savedmodel"}, default "tensorflow"
            Serialization format.
        **kwargs
            Additional arguments.

        Raises
        ------
        ValueError
            If the model is not fitted yet or unknown format is specified.
        """
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")

        if format == "tensorflow":
            # Save as Keras native format (.keras)
            if not filepath.endswith(".keras"):
                filepath += ".keras"
            self.model_.save(filepath)
        elif format == "h5":
            # Save as HDF5 format
            if not filepath.endswith(".h5"):
                filepath += ".h5"
            self.model_.save(filepath, save_format="h5")
        elif format == "savedmodel":
            # Save as TensorFlow SavedModel format
            self.model_.save(filepath, save_format="tf")
        else:
            raise ValueError(f"Unknown format {format}")

    def load(
        self, filepath: str, format: str = "tensorflow", **kwargs
    ) -> "TensorflowModel":
        """
        Load model parameters from disk.

        Parameters
        ----------
        filepath : str
            Source path.
        format : {"tensorflow", "h5", "savedmodel"}, default "tensorflow"
            Serialization format.
        **kwargs
            Additional arguments.

        Returns
        -------
        TensorflowModel
            The model instance with loaded weights.

        Raises
        ------
        ValueError
            If unknown format is specified.
        """
        if format == "tensorflow":
            # Load Keras native format
            if not filepath.endswith(".keras") and os.path.isfile(filepath + ".keras"):
                filepath += ".keras"
            self.model_ = K.models.load_model(filepath)
        elif format == "h5":
            # Load HDF5 format
            if not filepath.endswith(".h5") and os.path.isfile(filepath + ".h5"):
                filepath += ".h5"
            self.model_ = K.models.load_model(filepath)
        elif format == "savedmodel":
            # Load TensorFlow SavedModel format
            self.model_ = K.models.load_model(filepath)
        else:
            raise ValueError(f"Unknown format {format}")

        # Reinitialize optimizer after loading
        self.optimizer_ = self._get_optimizer()
        return self

    def freeze_layers(
        self,
        layer_names: Union[List[str], str] = None,
        parent_layer: Optional[Union[K.layers.Layer, str]] = None,
    ) -> None:
        """
        Freeze specified layers or all layers.

        Parameters
        ----------
        layer_names : list of str or str, optional
            Names of layers to freeze. If "all", freeze all layers.
            If None or empty list, no layers are frozen.
        parent_layer : keras.layers.Layer or str, optional
            Parent layer or layer name to search within for sub-layers.
            If provided, only layers within this parent will be considered.

        Raises
        ------
        ValueError
            If the model is not fitted yet or parent_layer not found.
        """
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")

        # Resolve parent layer if string name is provided
        search_layers = self.model_.layers
        if parent_layer is not None:
            if isinstance(parent_layer, str):
                parent_layer = self._find_layer_by_name(parent_layer)
                if parent_layer is None:
                    raise ValueError(f"Parent layer '{parent_layer}' not found.")
            search_layers = self._get_all_layers(parent_layer)

        # Handle "all" parameter
        if layer_names == "all":
            for layer in search_layers:
                layer.trainable = False
            return

        # Handle None or empty list
        if layer_names is None or len(layer_names) == 0:
            return

        # Ensure layer_names is a list
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        # Freeze specified layers
        for layer in search_layers:
            if layer.name in layer_names:
                layer.trainable = False

    def unfreeze_layers(
        self,
        layer_names: Union[List[str], str] = None,
        parent_layer: Optional[Union[K.layers.Layer, str]] = None,
    ) -> None:
        """
        Unfreeze specified layers or all layers.

        Parameters
        ----------
        layer_names : list of str or str, optional
            Names of layers to unfreeze. If "all", unfreeze all layers.
            If None or empty list, no layers are unfrozen.
        parent_layer : keras.layers.Layer or str, optional
            Parent layer or layer name to search within for sub-layers.
            If provided, only layers within this parent will be considered.

        Raises
        ------
        ValueError
            If the model is not fitted yet or parent_layer not found.
        """
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")

        # Resolve parent layer if string name is provided
        search_layers = self.model_.layers
        if parent_layer is not None:
            if isinstance(parent_layer, str):
                parent_layer = self._find_layer_by_name(parent_layer)
                if parent_layer is None:
                    raise ValueError(f"Parent layer '{parent_layer}' not found.")
            search_layers = self._get_all_layers(parent_layer)

        # Handle "all" parameter
        if layer_names == "all":
            for layer in search_layers:
                layer.trainable = True
            return

        # Handle None or empty list
        if layer_names is None or len(layer_names) == 0:
            return

        # Ensure layer_names is a list
        if isinstance(layer_names, str):
            layer_names = [layer_names]

        # Unfreeze specified layers
        for layer in search_layers:
            if layer.name in layer_names:
                layer.trainable = True

    def get_metrics(
        self,
        X=None,
        y=None,
        dataset=None,
        metrics: List[BaseMetric] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute metrics for given data.

        Parameters
        ----------
        X : array-like or tf.Tensor, optional
            Features. Ignored if ``dataset`` is provided.
        y : array-like or tf.Tensor, optional
            Labels. Ignored if ``dataset`` is provided.
        dataset : tf.data.Dataset, optional
            Pre-built dataset yielding ``(inputs, labels)``.
        metrics : list of BaseMetric
            Metrics to compute.
        batch_size : int, default 32
            Batch size when ``X`` and ``y`` are provided.
        device : str, optional
            Device to use for inference (currently not used in TensorFlow implementation).

        Returns
        -------
        dict
            Mapping from metric names to values.
        """
        if dataset is not None:
            X = dataset.map(lambda x, y: x)
            y = dataset.map(lambda x, y: y)
            y = tf.stack(list(y), axis=0).numpy()

        y_pred = self.predict(X, batch_size=batch_size)

        try:
            y_pred_proba = self.predict_proba(X, batch_size=batch_size)
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

        Parameters
        ----------
        **kwargs
            Additional arguments (ignored).

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
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "with_weight_init": self.with_weight_init,
        }

    def set_params(self, **kwargs) -> None:
        """
        Set initialization parameters for this model.

        Parameters
        ----------
        **kwargs
            Parameters to set on the instance.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """
        String representation of the wrapper and underlying class.

        Returns
        -------
        str
            String representation of the model.
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

    def _find_layer_by_name(self, layer_name: str) -> Optional[K.layers.Layer]:
        """
        Find a layer by name in the model.

        Parameters
        ----------
        layer_name : str
            Name of the layer to find.

        Returns
        -------
        keras.layers.Layer or None
            The layer if found, None otherwise.
        """
        for layer in self.model_.layers:
            if layer.name == layer_name:
                return layer
            # Check sub-layers recursively
            if hasattr(layer, "layers"):
                for sublayer in layer.layers:
                    found = self._find_layer_in_hierarchy(sublayer, layer_name)
                    if found is not None:
                        return found
        return None

    def _find_layer_in_hierarchy(
        self, layer: K.layers.Layer, layer_name: str
    ) -> Optional[K.layers.Layer]:
        """
        Recursively search for a layer by name in the layer hierarchy.

        Parameters
        ----------
        layer : keras.layers.Layer
            Current layer to check.
        layer_name : str
            Name of the layer to find.

        Returns
        -------
        keras.layers.Layer or None
            The layer if found, None otherwise.
        """
        if layer.name == layer_name:
            return layer
        if hasattr(layer, "layers"):
            for sublayer in layer.layers:
                found = self._find_layer_in_hierarchy(sublayer, layer_name)
                if found is not None:
                    return found
        return None

    def _get_all_layers(self, parent_layer: K.layers.Layer) -> List[K.layers.Layer]:
        """
        Recursively get all layers within a parent layer, including nested sub-layers.

        Parameters
        ----------
        parent_layer : keras.layers.Layer
            The parent layer to search within.

        Returns
        -------
        list of keras.layers.Layer
            All layers found within the parent layer, including the parent itself.
        """
        layers = [parent_layer]
        if hasattr(parent_layer, "layers"):
            for sublayer in parent_layer.layers:
                layers.extend(self._get_all_layers(sublayer))
        return layers

    def _create_dataset(self, X, y) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from features and labels.

        Parameters
        ----------
        X : array-like or tf.Tensor
            Input features.
        y : array-like or tf.Tensor
            Target labels.

        Returns
        -------
        tf.data.Dataset
            Dataset containing the input data.
        """
        return tf.data.Dataset.from_tensor_slices((X, y))

    def clone(self) -> "TensorflowModel":
        """
        Create a shallow clone with the same initialization parameters.

        Returns
        -------
        TensorflowModel
            New instance with copied parameters.
        """
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
        """
        Get the recorded metric history.

        Parameters
        ----------
        split : {"train", "val"}, default "train"
            Which split to return history for. Validation history is
            recorded at epoch granularity.

        Returns
        -------
        dict
            Mapping from metric name to list of values in time order.

        Raises
        ------
        ValueError
            If split is not "train" or "val".
        """
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
        """
        Check if the underlying model supports loss history.

        Returns
        -------
        bool
            True if loss history is available, False otherwise.
        """
        return False


class _PeriodicSaveCallback(K.callbacks.Callback):
    """
    Internal callback for saving model checkpoints at regular intervals.

    Parameters
    ----------
    save_every_n_epochs : int
        Save model every n epochs.
    save_folder : str
        Directory path where checkpoints will be saved.
    save_format : str
        Serialization format for checkpoints.
    """

    def __init__(self, save_every_n_epochs, save_folder, save_format):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self.save_folder = save_folder
        self.save_format = save_format

    def on_epoch_end(self, epoch, logs=None):
        """
        Save model checkpoint at the end of specified epochs.

        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed).
        logs : dict, optional
            Dictionary of logs from the current epoch.
        """
        if (epoch + 1) % self.save_every_n_epochs == 0:
            epoch_filepath = os.path.join(self.save_folder, f"model_epoch_{epoch + 1}")
            try:
                if self.save_format == "tensorflow":
                    if not epoch_filepath.endswith(".keras"):
                        epoch_filepath += ".keras"
                    self.model.save(epoch_filepath)
                elif self.save_format == "h5":
                    if not epoch_filepath.endswith(".h5"):
                        epoch_filepath += ".h5"
                    self.model.save(epoch_filepath, save_format="h5")
                elif self.save_format == "savedmodel":
                    self.model.save(epoch_filepath, save_format="tf")
                else:
                    raise ValueError(f"Unknown format {self.save_format}")
                print(f"Model saved at epoch {epoch + 1}: {epoch_filepath}")
            except Exception as e:
                print(f"Failed to save model at epoch {epoch + 1}: {e}")
