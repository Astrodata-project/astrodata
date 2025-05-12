from astrodata.data.loaders.base import BaseLoader
from astrodata.data.processors.base import AbstractPreprocessor
from astrodata.data.schemas import ProcessedData
from astrodata.data.utils import convert_to_processed_data


class DataPipeline:
    """
    A pipeline for processing data using a loader and a series of processors.

    Attributes:
        loader (BaseLoader): The data loader responsible for loading raw data.
        processors (list[AbstractPreprocessor]): A list of processors to process the data.

    Methods:
        run(path: str) -> ProcessedData:
            Executes the pipeline by loading data from the given path,
            applying the processors sequentially, and converting the
            result into a ProcessedData object.
    """

    def __init__(
        self,
        loader: BaseLoader,
        processors: list[AbstractPreprocessor],
    ):
        self.loader = loader
        self.preprocessor = processors

    def run(self, path: str) -> ProcessedData:
        """
        Executes the data pipeline.

        Args:
            path (str): The file path to load the raw data from.

        Returns:
            ProcessedData: The processed data after applying all processors.
        """
        data = self.loader.load(path)
        for preprocessor in self.preprocessor:
            data = preprocessor.preprocess(data)
        return convert_to_processed_data(data)
