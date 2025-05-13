from abc import ABC, abstractmethod
import pickle
from typing import Any, Optional
from astrodata.preml.schemas import Premldata


class AbstractProcessor(ABC):
    """
    An abstract base class for preml processors.

    Subclasses must implement the `process` method to define how
    the input `Premldata` is processed.

    Methods:
        process(preml: Premldata) -> Premldata:
            Abstract method to process the input `Premldata` and return
            a new `Premldata` object.
    """

    def __init__(self, **kwargs: Any):
        """
        Initializes the processor with an empty dictionary to store artifacts.
        """
        self.artifact = None
        self.kwargs = kwargs

    def save_artifact(self, artifact: Any, path: str):
        """
        Saves an artifact to a specified path.

        Args:
            name (str): The name of the artifact to save.
            path (str): The path where the artifact should be saved.
        """
        self.artifact = artifact
        with open(path, "wb") as f:
            pickle.dump(self.artifact, f)

    def load_artifact(self, path: str):
        """
        Loads an artifact from a specified path.

        Args:
            name (str): The name of the artifact to load.
            path (str): The path from where the artifact should be loaded.
        """
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
