import os
import pickle
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import trange

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel


class PytorchModel(BaseMlModel):
    """
    A lightweight wrapper around PyTorch models providing a unified
    training/prediction interface and metric tracking.

    Notes
    -----
    This class expects a callable ``loss_fn`` (e.g., ``nn.CrossEntropyLoss``)
    and an ``optimizer`` class (e.g., ``optim.AdamW``). ``model_class`` can be
    either an instantiated ``nn.Module`` or a class to be constructed from
    ``model_params``.
    """

    def __init__(
        self,
        model_class: Module = None,
        loss_fn: Optional[Any] = None,
        optimizer: Optional[Optimizer] = None,
        model_params: Optional[Dict] = None,
        optimizer_params: Optional[Dict] = None,
        device: Optional[str] = None,
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
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_ = None if not with_weight_init else self._get_model()
        self.optimizer_ = None
        self.loss_fn_ = None
        self.metrics_history_ = None
        self._val_metrics_history_ = None

    def fit(
        self,
        X: Optional[Any] = None,
        y: Optional[Any] = None,
        dataset: Optional[Dataset] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        fine_tune: bool = False,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
        dataset_val: Optional[Dataset] = None,
        save_every_n_epochs: Optional[int] = None,
        save_folder: Optional[str] = None,
        save_format: str = "torch",
        shuffle: bool = True,
        seed: int = 42,
        **kwargs,
    ) -> "PytorchModel":
        """
        Fit the model using the provided data or dataset.

        Parameters
        ----------
        X : array-like or torch.Tensor, optional
            Training features. Ignored if ``dataset`` is provided.
        y : array-like or torch.Tensor, optional
            Training labels. Ignored if ``dataset`` is provided.
        dataset : torch.utils.data.Dataset, optional
            Pre-built training dataset yielding ``(inputs, labels)``.
        epochs : int, optional
            Number of training epochs. Defaults to the instance value.
        batch_size : int, optional
            Batch size for training/prediction. Defaults to the instance value.
        device : str, optional
            Device to use (e.g., ``"cuda"`` or ``"cpu"``). Defaults to instance.
        metrics : list of BaseMetric, optional
            Metrics to compute during training (per step) and validation (per epoch).
        fine_tune : bool, default False
            If True, reuse existing model weights/optimizer state when available.
        X_val : array-like or torch.Tensor, optional
            Validation features for epoch-wise metric tracking.
        y_val : array-like or torch.Tensor, optional
            Validation labels for epoch-wise metric tracking.
        save_every_n_epochs : int, optional
            Save model every n epochs if provided.
        save_folder : str, optional
            Directory path where checkpoints will be saved.
        save_format : {"torch","pkl","safetensors"}, default "torch"
            Serialization format for checkpoints.

        Returns
        -------
        PytorchModel
            The fitted model instance.
        """

        epochs = epochs if epochs is not None else self.epochs
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
            dataset = self._create_dataset(X, y, device)
        # if a dataloader was passed, we trust it yields tensors on the right device

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        if dataset_val is None and X_val is not None and y_val is not None:
            dataset_val = self._create_dataset(X_val, y_val, device)

        if not fine_tune or self.model_ is None:
            self.model_ = self._get_model().to(self.device)
            self.optimizer_ = self._get_optimizer(self.model_)
        self.loss_fn_ = self.loss_fn()

        self.model_.train()

        with trange(epochs, desc="Epochs", position=2) as t:
            for epoch in t:
                last_loss = self._train_one_epoch(
                    optimizer=self.optimizer_,
                    epoch_index=epoch,
                    model=self.model_,
                    loss_fn=self.loss_fn_,
                    training_loader=dataloader,
                    metrics=metrics,
                    dataset_val=dataset_val,
                    batch_size=batch_size,
                    device=device,
                )
                t.set_postfix({"last_loss": f"{last_loss:.4f}"})
                # periodic checkpoint
                if (
                    save_every_n_epochs
                    and save_folder
                    and (epoch + 1) % save_every_n_epochs == 0
                ):
                    # map format to extension
                    ext_map = {
                        "torch": "pt",
                        "pkl": "pkl",
                        "safetensors": "safetensors",
                    }
                    ext = ext_map.get(save_format, save_format)
                    fname = f"checkpoint_{epoch+1}.{ext}"
                    os.makedirs(save_folder, exist_ok=True)
                    path = os.path.join(save_folder, fname)
                    self.save(path, format=save_format)
        return self

    def _train_one_epoch(
        self,
        optimizer: Optimizer,
        epoch_index: int,
        model: Module,
        loss_fn,
        training_loader: torch.utils.data.DataLoader,
        metrics: Optional[List[BaseMetric]] = None,
        dataset_val: Optional[Dataset] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> float:

        for i, data in enumerate(training_loader):
            # Each data instance is an input + label pair
            inputs, labels = data

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Backward and optimize
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # Compute predictions for metrics without toggling eval mode
        with torch.no_grad():
            if outputs.dim() > 1 and outputs.shape[-1] > 1:
                preds = outputs.detach().argmax(dim=1).cpu().numpy()
            else:
                preds = outputs.detach().cpu().squeeze().numpy()

        if metrics is not None:
            y_true = labels.detach().cpu().numpy()
            for metric in metrics:
                self.metrics_history_.append(
                    (f"{metric.get_name()}_epoch", metric(y_true, preds))
                )
            self.metrics_history_.append(("loss_epoch", loss.item()))

        # Epoch-level validation metrics
        if metrics is not None and dataset_val is not None:
            val_scores = self.get_metrics(
                dataset=dataset_val,
                metrics=metrics,
                batch_size=batch_size or self.batch_size or 32,
                device=device or self.device,
            )
            for name, value in val_scores.items():
                self._val_metrics_history_.append((f"{name}_epoch", value))

            val_loss_acc = []
            for i, data in enumerate(
                DataLoader(
                    dataset_val,
                    batch_size=batch_size or self.batch_size or 32,
                    shuffle=False,
                )
            ):
                inputs, labels = data

                # Forward pass
                outputs = model(inputs)
                val_loss_acc.append(loss_fn(outputs, labels).item())

            self._val_metrics_history_.append(
                ("loss_epoch", sum(val_loss_acc) / len(val_loss_acc))
            )

        return loss.item()

    def predict(
        self, data, batch_size: int, device: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Predict outputs for input ``X``.

        Parameters
        ----------
        data : array-like, torch.Tensor, or Dataset
            Features to predict. If a Dataset is provided, it should yield
            feature tensors only.
        batch_size : int
            Batch size.
        device : str, optional
            Device to use for inference. Defaults to instance device.

        Returns
        -------
        numpy.ndarray
            Predicted labels for classification, or raw outputs for regression.

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        """
        return self._predict(data, batch_size, device, use_proba=False)

    def predict_proba(
        self, data, batch_size: int, device: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Predict class probabilities for input ``X``.

        Parameters
        ----------
        data : array-like, torch.Tensor, or Dataset
            Features to predict. If a Dataset is provided, it should yield
            feature tensors only.
        batch_size : int
            Batch size used when ``X`` is not a Dataset.
        device : str, optional
            Device to use for inference. Defaults to instance device.

        Returns
        -------
        numpy.ndarray
            Predicted probabilities with shape ``[N, n_classes]``.

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        """
        return self._predict(data, batch_size, device, use_proba=True)

    def _predict(
        self, X, batch_size: int, device: Optional[str], use_proba: bool
    ) -> Any:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")

        device = device or self.device
        self.model_.eval()

        if not isinstance(X, Dataset):
            X = torch.tensor(X, dtype=torch.float32)
            X = TensorDataset(X)

        dataloader = DataLoader(dataset=X, batch_size=batch_size, shuffle=False)
        outputs = []

        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X[0].to(device)
                out = self.model_(batch_X)
                try:
                    outputs.append(out.cpu())
                except AttributeError:
                    outputs.append(out)

        outputs = torch.cat(outputs, dim=0)

        if use_proba:
            if outputs.shape[-1] == 1:
                probs = torch.sigmoid(outputs)
                probs = torch.cat([1 - probs, probs], dim=1)  # shape: [N, 2]
            else:
                probs = F.softmax(outputs, dim=1)
            return probs.numpy()

        if outputs.shape[-1] == 1:
            return outputs.numpy().ravel()
        return outputs.argmax(dim=1).numpy()

    def score(
        self,
        dataset=None,
        X=None,
        y=None,
        device=None,
        batch_size=None,
    ):
        """
        Compute the average loss on the given data.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset, optional
            Pre-built dataset yielding ``(inputs, labels)``.
        X : array-like or torch.Tensor, optional
            Features. Ignored if ``dataset`` is provided.
        y : array-like or torch.Tensor, optional
            Labels. Ignored if ``dataset`` is provided.
        device : str, optional
            Device to use for inference. Defaults to instance device.
        batch_size : int, optional
            Batch size when ``X`` and ``y`` are provided.

        Returns
        -------
        float
            Average loss over the dataset.

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        ValueError
            If neither dataset nor (X, y) are provided.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")

        if dataset is None and (X is None or y is None):
            raise ValueError("Either dataset or both X and y must be provided.")

        if dataset is None and (X is not None and y is not None):
            dataset = self._create_dataset(X, y, device=device)

        dataloader = DataLoader(
            dataset, batch_size=batch_size or self.batch_size or 32, shuffle=False
        )

        device = device or self.device
        self.model_.eval()
        loss_fn = self.loss_fn()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model_(inputs)
                loss = loss_fn(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def get_scorer_metric(self):
        pass

    def save(self, filepath: str, format: str = "torch", **kwargs) -> None:
        """
        Save the model parameters and optimizer state.

        Parameters
        ----------
        filepath : str
            Destination path.
        format : {"torch", "pkl", "safetensors"}
            Serialization format.
        """
        state = {
            "model_state_dict": self.model_.state_dict(),
            "optimizer_state_dict": self.optimizer_.state_dict(),
            "model_params": self.model_params,
        }

        if format == "torch":
            torch.save(state, filepath)
        elif format == "pkl":
            with open(filepath, "wb") as f:
                pickle.dump(state, f)
        elif format == "safetensors":
            safetensors_save(self.model_.state_dict(), filepath)
        else:
            raise ValueError(f"Unknown format {format}")

    def load(self, filepath: str, format: str = "torch", **kwargs) -> "PytorchModel":
        """
        Load model parameters and optimizer state from disk.

        Parameters
        ----------
        filepath : str
            Source path.
        format : {"torch", "pkl", "safetensors"}
            Serialization format.

        Returns
        -------
        PytorchModel
            The model instance with loaded weights.
        """
        if format == "torch":
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model_ = self._get_model().to(self.device)
            self.model_.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer_ = self._get_optimizer(self.model_)
            self.optimizer_.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model_params = checkpoint.get("model_params", {})
        elif format == "pkl":
            with open(filepath, "rb") as f:
                checkpoint = pickle.load(f)
            self.model_ = self._get_model().to(self.device)
            self.model_.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer_ = self._get_optimizer(self.model_)
            self.optimizer_.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model_params = checkpoint.get("model_params", {})
        elif format == "safetensors":
            state_dict = safetensors_load(filepath)
            self.model_ = self._get_model().to(self.device)
            self.model_.load_state_dict(state_dict)
            self.optimizer_ = self._get_optimizer(self.model_)
        else:
            raise ValueError(f"Unknown format {format}")
        return self

    def freeze_layers(self, layer_names: List[str]) -> None:
        """
        Freeze all layers except those included in ``layer_names``.

        Parameters
        ----------
        layer_names : list of str
            Names of modules to unfreeze.
        """
        # freeze all
        for param in self.model_.parameters():
            param.requires_grad = False
        # unfreeze selected
        for name, module in self.model_.named_modules():
            if name in layer_names:
                for param in module.parameters():
                    param.requires_grad = True

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
        X : array-like or torch.Tensor, optional
            Features. Ignored if ``dataset`` is provided.
        y : array-like or torch.Tensor, optional
            Labels. Ignored if ``dataset`` is provided.
        dataset : torch.utils.data.Dataset, optional
            Pre-built dataset yielding ``(inputs, labels)``.
        metrics : list of BaseMetric
            Metrics to compute.
        batch_size : int, default 32
            Batch size when ``X`` and ``y`` are provided.
        device : str, optional
            Device to use for inference.

        Returns
        -------
        dict
            Mapping from metric names to values.

        Raises
        ------
        ValueError
            If neither dataset nor (X, y) are provided.
        """
        if (X is None or y is None) and dataset is None:
            raise ValueError("Either dataset or both X and y must be provided.")

        if dataset is None:
            dataset = self._create_dataset(X, y, device=device)

        dataloader = DataLoader(
            dataset, batch_size=batch_size or self.batch_size or 32, shuffle=False
        )

        # Collect all predictions and true labels
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []

        device = device or self.device
        self.model_.eval()

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Get predictions
                outputs = self.model_(inputs)

                # Convert to predictions
                if outputs.dim() > 1 and outputs.shape[-1] > 1:
                    preds = outputs.argmax(dim=1)
                    # Get probabilities
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                else:
                    preds = outputs.squeeze()
                    # Get probabilities for binary case
                    probs = torch.sigmoid(outputs)
                    if probs.dim() == 1:
                        probs = torch.stack([1 - probs, probs], dim=1)

                # Collect results
                all_y_true.append(labels.cpu().numpy())
                all_y_pred.append(preds.cpu().numpy())
                all_y_pred_proba.append(probs.cpu().numpy())

        # Concatenate all results
        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)
        y_pred_proba = np.concatenate(all_y_pred_proba)

        results = {}
        for metric in metrics:
            try:
                score = metric(y_true, y_pred_proba)
            except (ValueError, TypeError):
                score = metric(y_true, y_pred)
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
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
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
        """
        return f"{self.__class__.__name__}(torch_model={self.model_class.__class__.__name__})"

    def _get_model(self):
        if isinstance(self.model_class, Module):
            return self.model_class
        else:
            return self.model_class(**(self.model_params or {}))

    def _get_optimizer(self, model):
        if isinstance(self.optimizer, Optimizer):
            return self.optimizer
        else:
            return self.optimizer(model.parameters(), **(self.optimizer_params or {}))

    def _create_dataset(self, X, y, device: Optional[str] = None) -> Dataset:
        """
        Create a Dataset from X and y tensors.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input features.
        y : array-like or torch.Tensor
            Target labels.
        batch_size : int
            Batch size for the Dataset.
        device : str, optional
            Device to place tensors on. Defaults to self.device.
        shuffle : bool, default True
            Whether to shuffle the data.

        Returns
        -------
        Dataset
            A Dataset containing the input data.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        if device is None:
            device = self.device

        dataset = TensorDataset(X.to(device), y.to(device))
        return dataset

    def clone(self) -> "PytorchModel":
        """
        Create a shallow clone with the same initialization parameters.

        Returns
        -------
        PytorchModel
            New instance with copied parameters.
        """

        new_instance = self.__class__(
            model_class=self.model_class,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            model_params=self.model_params,
            optimizer_params=self.optimizer_params,
            device=self.device,
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
        """
        history = (
            self.metrics_history_
            if split == "train"
            else self._val_metrics_history_ if split == "val" else None
        )
        d = {}
        if history is None:
            return d
        for x, y in history:
            d.setdefault(x, []).append(y)
        return d

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
