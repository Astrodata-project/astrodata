from abc import ABC, abstractmethod


class BaseModelSelector(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        pass

    @abstractmethod
    def get_best_model(self):
        pass

    @abstractmethod
    def get_best_params(self):
        pass

    def get_params(self, **kwargs) -> dict:
        pass
