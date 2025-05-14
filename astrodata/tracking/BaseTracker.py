from abc import ABC, abstractmethod

class BaseTracker(ABC):
    @abstractmethod
    def wrap_fit(self, obj):
        pass
