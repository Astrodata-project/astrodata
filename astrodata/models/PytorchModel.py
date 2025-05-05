import torch
from BaseModel import BaseModel
from torch.utils.data import TensorDataset, DataLoader
from typing import Type, Optional

class PyTorchModel(BaseModel):
    model: Optional[torch.nn.Module]
    optimizer: Optional[torch.optim.Optimizer]
    criterion: Optional[torch.nn.Module]  # Typically a loss module

    def __init__(self):
        super().__init__()
        self.model = None
        self.optimizer = None
        self.criterion = None

    def config(
        self,
        model_class: Type[torch.nn.Module],
        optimizer_class: Type[torch.optim.Optimizer],
        criterion: torch.nn.Module,
        model_kwargs: Optional[dict] = None,
        optimizer_kwargs: Optional[dict] = None
    ) -> None:
        model_kwargs = model_kwargs if model_kwargs is not None else {}
        optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

        self.model = model_class(**model_kwargs)
        self.criterion = criterion
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)

    def fit(self, X, y, epochs=10, batch_size=32):
        # Convert np.ndarray to torch.Tensor
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            for xb, yb in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X = torch.from_numpy(X).float()
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
        return pred.numpy()