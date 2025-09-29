import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional

from astrodata.preml.schemas import Premldata


class PremlProcessor(ABC):
    """
    An abstract base class for preml processors.

    Subclasses must implement the `process` method to define how
    the input `Premldata` is processed.

    Methods:
        process(preml: Premldata) -> Premldata:
            Abstract method to process the input `Premldata` and return
            a new `Premldata` object.
    """

    def __init__(self, artifact_path: Optional[str] = None, **kwargs: Any):
        """
        Initializes the processor with an optional artifact path and keyword arguments.

        Args:
            artifact_path (Optional[str]): Path to save/load processor artifacts.
            **kwargs: Additional keyword arguments for processor configuration.
        """
        # TODO: Short term solution to save artifacts, need to think of a better way to handle this
        self.save_path = artifact_path
        self.artifact = self.load_artifact(self.save_path)
        self.kwargs = kwargs

    def save_artifact(self, artifact: Any):
        """
        Saves an artifact to a specified path.

        Args:
            Artifact (Any): The artifact to be saved, which can be any object.
        """
        self.artifact = artifact
        if self.save_path is None:
            return
        with open(self.save_path, "wb") as f:
            pickle.dump(self.artifact, f)

    def load_artifact(self, path: str):
        """
        Loads an artifact from a specified path.

        Args:
            path (str): The path from where the artifact should be loaded.
        """
        if path is None:
            return None
        try:
            with open(path, "rb") as f:
                self.artifact = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Artifact not found at '{path}'.")

    @abstractmethod
    def process(
        self, preml: Premldata, artifact: Optional[str] = None, **kwargs
    ) -> Premldata:
        """process the input Premldata and returns a new Premldata object."""
        pass
