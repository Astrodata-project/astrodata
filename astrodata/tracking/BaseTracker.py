from abc import ABC, abstractmethod


class BaseTracker(ABC):
    @abstractmethod
    def wrap_fit(self, obj):
        pass

    @abstractmethod
    def register_best_model(
        self,
        metric,
        model_artifact_path="model",
        registered_model_name=None,
        split_name="train",
        stage="Production",
    ):
        pass
