import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.models.BaseMlModel import BaseMlModel


class PytorchModel(BaseMlModel):
    def __init__(
        self,
        model_class: Module,
        loss_fn: Optional[Any] = None,
        optimizer: Optional[Optimizer] = None,
        model_params: Optional[Dict] = None,
        optimizer_params: Optional[Dict] = None,
        device: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        random_state: int = random.randint(0, 2**32),
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

        self.model_ = None
        self.optimizer_ = None
        self.loss_fn_ = None
        self.metrics_history_ = None

    def fit(
        self,
        X,
        y,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
        metrics: Optional[List[BaseMetric]] = None,
        **kwargs,
    ) -> "PytorchModel":

        epochs = epochs if epochs is not None else self.epochs
        batch_size = batch_size if batch_size is not None else self.batch_size
        self.metrics_history_ = []

        if epochs <= 0:
            raise ValueError("Number of epochs must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        if device is None:
            device = self.device

        dataset = TensorDataset(X.to(device), y.to(device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model_ = self._get_model().to(self.device)
        self.optimizer_ = self._get_optimizer(self.model_)
        self.loss_fn_ = self.loss_fn()

        self.model_.train()

        with trange(epochs, desc="Epochs", position=2) as t:
            for epoch in t:
                last_loss = self._train_one_epoch(
                    epoch_index=epoch,
                    optimizer=self.optimizer_,
                    model=self.model_,
                    loss_fn=self.loss_fn_,
                    training_loader=dataloader,
                    metrics=metrics,
                )
                t.set_postfix({"last_loss": f"{last_loss:.4f}"})
        return self

    def predict(
        self, X, batch_size: int, device: Optional[str] = None, **kwargs
    ) -> Any:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")

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

    def predict_proba(
        self, X, batch_size: int, device: Optional[str] = None, **kwargs
    ) -> Any:
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet.")

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
            # Binary classification with single output: use sigmoid and stack probabilities
            probs = torch.sigmoid(outputs)
            probs = torch.cat([1 - probs, probs], dim=1)  # shape: [N, 2]
            return probs.numpy()
        else:
            # Multi-class: use softmax
            probs = F.softmax(outputs, dim=1)
            return probs.numpy()

    def score(self):
        pass

    def get_scorer_metric(self):
        pass

    def save(self, filepath: str, **kwargs) -> None:
        torch.save(
            {
                "model_state_dict": self.model_class.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_params": self.model_params,
                # Add any extra info as needed
            },
            filepath,
        )

    def load(self, filepath: str, **kwargs) -> "PytorchModel":
        checkpoint = torch.load(filepath)
        self.model_class.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model_params = checkpoint.get("model_params", {})
        self.model_ = self.model_class
        return self

    def get_metrics(
        self,
        X,
        y,
        metrics: List[BaseMetric] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        y_pred = self.predict(X, batch_size, device)
        try:
            y_pred_proba = self.predict_proba(X, batch_size, device)
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
        params = {"model_class": self.model_class}
        params["loss_fn"] = self.loss_fn
        params["optimizer"] = self.optimizer
        params["model_params"] = self.model_params
        params["optimizer_params"] = self.optimizer_params
        params["device"] = self.device
        params["epochs"] = self.epochs
        params["batch_size"] = self.batch_size
        params["random_state"] = self.random_state

        return params

    def set_params(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(torch_model={self.model_class.__class__.__name__})"

    def _train_one_epoch(
        self,
        optimizer: Optimizer,
        epoch_index: int,
        model: Module,
        loss_fn,
        training_loader: torch.utils.data.DataLoader,
        metrics: Optional[List[BaseMetric]] = None,
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
            
            predictions = self.predict(inputs, batch_size=self.batch_size, device=self.device) 
            
            for metric in metrics:
                self.metrics_history_.append((f"{metric.get_name()}_step", metric(labels.cpu(), predictions)))
                self.metrics_history_.append(("loss_step", loss.item()))

        return loss.item()

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

    def clone(self) -> "PytorchModel":

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
    
    def get_metrics_history(self):
        d = {}
        for x, y in self.metrics_history_:
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
