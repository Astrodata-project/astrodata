from astrodata.data.loaders.base import BaseLoader
from astrodata.data.transformers.base import AbstractTransformer
from astrodata.data.schemas import ProcessedData
from astrodata.data.utils import convert_to_processed_data


class DataPipeline:
    """
    A pipeline for processing data using a loader and a series of transformers.

    Attributes:
        loader (BaseLoader): The data loader responsible for loading raw data.
        transformers (list[AbstractTransformer]): A list of transformers to process the data.

    Methods:
        run(path: str) -> ProcessedData:
            Executes the pipeline by loading data from the given path,
            applying the transformers sequentially, and converting the
            result into a ProcessedData object.
    """

    def __init__(self, loader: BaseLoader, transformers: list[AbstractTransformer]):
        self.loader = loader
        self.transformer = transformers

    def run(self, path: str) -> ProcessedData:
        """
        Executes the data pipeline.

        Args:
            path (str): The file path to load the raw data from.

        Returns:
            ProcessedData: The processed data after applying all transformers.
        """
        data = self.loader.load(path)
        for transformer in self.transformer:
            data = transformer.transform(data)
        return convert_to_processed_data(data)
