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
        """Return a new instance with the same configuration (but not fitted)."""
        return self.__class__(model_class=self.model_class, **self.model_params)
    
    def get_metrics(self, X_test, y_test, metric_classes:List[BaseMetric]=None):
        y_pred = self.predict(X_test)
        results = {}
        for metric_cls in metric_classes:
            metric = metric_cls()
            score = metric(y_test, y_pred)
            results[metric.get_name()] = score
        return results
