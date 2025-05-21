from abc import ABC, abstractmethod
from typing import List

from astrodata.ml.metrics.BaseMetric import BaseMetric


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass

    @abstractmethod
    def score(self, X, y, **kwargs):
        pass

    @abstractmethod
    def save(self, filepath, **kwargs):
        pass

    @abstractmethod
    def load(self, filepath, **kwargs):
        pass

    @abstractmethod
    def get_metrics(self, X_test, y_test, **kwargs):
        pass

    def get_params(self, **kwargs):
        # Optional: Only meaningful for models that have hyperparameters
        raise NotImplementedError

    def set_params(self, **kwargs):
        # Optional: Only meaningful for models that have hyperparameters
        raise NotImplementedError

    def clone(self):
        new_instance = self.__class__(model_class=self.model_class, **self.model_params)
        # Copy over any decorated methods from self's __dict__ to the new instance
        for attr, value in self.__dict__.items():
            if callable(value):
                setattr(new_instance, attr, value)
        return new_instance

    def get_metrics(self, X_test, y_test, metrics: List[BaseMetric] = None):
        y_pred = self.predict(X_test)
        results = {}
        for metric in metrics:
            score = metric(y_test, y_pred)
            results[metric.get_name()] = score
        return results
