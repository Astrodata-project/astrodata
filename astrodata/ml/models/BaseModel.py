from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass

    @abstractmethod
    def save(self, filepath, **kwargs):
        pass

    @abstractmethod
    def load(self, filepath, **kwargs):
        pass

    def get_params(self, **kwargs):
        # Optional: Only meaningful for models that have hyperparameters
        raise NotImplementedError

    def set_params(self, **kwargs):
        # Optional: Only meaningful for models that have hyperparameters
        raise NotImplementedError
    
    def clone(self):
        """Return a new instance with the same configuration (but not fitted)."""
        return self.__class__(model_class=self.model_class, **self.model_params)
