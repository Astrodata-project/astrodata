from abc import ABC, abstractmethod
from astrodata.data.schemas import RawData


class BaseLoader(ABC):
    """
    Abstract base class for data loaders.

    This class defines the interface for all data loaders, requiring
    the implementation of the `load` method to load data from a given path.

    Methods:
        load(path: str) -> RawData:
            Abstract method to load data from the specified path.
    """

    @abstractmethod
    def load(self, path: str) -> RawData:
        pass
