import random
from typing import Any, Dict, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel


class PytorchModel(BaseMlModel):
    """
    A scikit-learn-like wrapper for PyTorch models.
    """

    def __init__(
        self,
        torch_model: Module,
        loss_fn,
        optimizer: Optimizer,
        device: Optional[str] = None,
        random_state: int = random.randint(0, 2**32),
    ):
        """
        Initialize the PytorchModel.

        Args:
            torch_model (Module): PyTorch neural network model.
            loss_fn: Loss function for optimization.
            optimizer (Optimizer): Optimizer instance.
            device (str or None): Device to use ('cuda' or 'cpu').
            random_state (int): Random seed.
        """
        self.random_state = random_state
        self.torch_model = torch_model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = (
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_params = torch_model.parameters()
        self.model_ = None  # For scikit-learn compatibility
        super().__init__()

    def fit(
        self,
        X,
        y,
        epochs: int,
        batch_size: int,
        device: Optional[str] = None,
        **kwargs,
    ) -> "PytorchModel":
        """
        Fit the PyTorch model on the data.

        Args:
            X: Input features.
            y: Target labels.
            epochs (int): Number of epochs.
            batch_size (int): Mini-batch size.
            device (str or None): Device to use.

        Returns:
            self
        """
        if epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X.to(device), y.to(device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model_ = self.torch_model
        self.model_.to(self.device)
        self.model_.train()

        with trange(epochs, desc="Epochs") as t:
            for epoch in t:
                last_loss = self._train_one_epoch(
                    epoch_index=epoch,
                    optimizer=self.optimizer,
                    model=self.model_,
                    loss_fn=self.loss_fn,
                    training_loader=dataloader,
                )
                t.set_postfix({"last_loss": f"{last_loss:.4f}"})
        return self

    def predict(
        self, X, batch_size: int, device: Optional[str] = None, **kwargs
    ) -> Any:
        """
        Predict using the trained model.

        Args:
            X: Input features.
            batch_size (int): Mini-batch size.
            device (str or None): Device to use.

        Returns:
            Predictions as a numpy array.
        """
        if device is None:
            device = self.device

        self.model_.eval()

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        dataloader = DataLoader(X, batch_size=batch_size, shuffle=False)
        outputs = []

        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(device)
                out = self.model_(batch_X)
                outputs.append(out.cpu())

        outputs = torch.cat(outputs, dim=0)

        if outputs.shape[-1] == 1:
            return outputs.numpy().ravel()
        else:
            return outputs.argmax(dim=1).numpy()

    def score(self):
        pass

    def get_scorer_metric(self):
        pass

    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the model and optimizer state.

        Args:
            filepath (str): Path to save the model.
        """
        torch.save(
            {
                "model_state_dict": self.torch_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_params": self.model_params,
                # Add any extra info as needed
            },
            filepath,
        )

    def load(self, filepath: str, **kwargs) -> "PytorchModel":
        """
        Load the model and optimizer state.

        Args:
            filepath (str): Path to the saved model.

        Returns:
            self
        """
        checkpoint = torch.load(filepath)
        self.torch_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_params = checkpoint.get("model_params", {})
        self.model_ = self.torch_model
        return self

    def get_metrics(self):
        pass

    def get_params(self, **kwargs) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            dict: Model parameters.
        """
        params = {"model_class": self.torch_model}
        params["random_state"] = self.random_state
        params["loss"] = self.loss_fn
        params["optimizer"] = self.optimizer
        params["device"] = self.device
        params.update(self.model_params)

        return params

    def set_params(self, **kwargs) -> None:
        """
        Set model parameters.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.model_params.update(kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(torch_model={self.torch_model.__class__.__name__})"

    def _train_one_epoch(
        self,
        epoch_index: int,
        optimizer: Optimizer,
        model: Module,
        loss_fn,
        training_loader: torch.utils.data.DataLoader,
    ) -> float:
        """
        Train for one epoch.

        Returns:
            float: Last batch loss.
        """
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

        return loss.item()
