from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def config(self, *args, **kwargs):
        """Set up model, optimizer, loss, etc."""
        pass

    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs):
        """Predict using the model."""
        pass

    @abstractmethod
    def save(self, filepath, *args, **kwargs):
        """Save the model."""
        pass

    @abstractmethod
    def load(self, filepath, *args, **kwargs):
        """Load the model."""
        pass
