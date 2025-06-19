from abc import ABC, abstractmethod


class BaseMetric(ABC):
    @abstractmethod
    def __init__(self, metric, **kwargs):
        pass

    @abstractmethod
    def __call__(self, y_true, y_pred, **kwargs):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @property
    @abstractmethod
    def greater_is_better(self):
        pass

    def __eq__(self, other):
        if name := getattr(other, "get_name", None):
            return self.get_name() == name()
