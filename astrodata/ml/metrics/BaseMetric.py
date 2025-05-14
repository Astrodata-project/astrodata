from abc import ABC, abstractmethod

class BaseMetric(ABC):
    @abstractmethod
    def __call__(self, y_true, y_pred, **kwargs):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def clone(self):
        return self.__class__(**self.get_params())
