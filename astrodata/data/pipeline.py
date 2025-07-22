from pathlib import Path

from astrodata.data.loaders.base import BaseLoader
from astrodata.data.processors.base import AbstractProcessor
from astrodata.data.schemas import ProcessedData
from astrodata.data.utils import convert_to_processed_data
from astrodata.utils.utils import get_output_path, read_config


class DataPipeline:
    """
    A pipeline for processing data using a loader and a series of processors.

    Attributes:
        loader (BaseLoader): The data loader responsible for loading raw data.
        processors (list[AbstractProcessor]): A list of processors to process the data.

    Methods:
        run(path: str) -> ProcessedData:
            Executes the pipeline by loading data from the given path,
            applying the processors sequentially, and converting the
            result into a ProcessedData object.
    """

    def __init__(
        self, config_path: str, loader: BaseLoader, processors: list[AbstractProcessor]
    ):
        self.config = read_config(config_path)
        self.loader = loader
        self.processor = processors

    def run(self, path: str, dump_output: bool = True) -> ProcessedData:
        """
        Executes the data pipeline.

        Args:
            path (str): The file path to load the raw data from.

        Returns:
            ProcessedData: The processed data after applying all processors.
        """
        data = self.loader.load(self.config["project_path"] / path)
        for processor in self.processor:
            data = processor.process(data)

        processed_data = convert_to_processed_data(data)
        if dump_output:
            output_path = get_output_path(
                self.config["project_path"], "processed_data", "processed_data.parquet"
            )
            processed_data.dump_parquet(output_path)
        return processed_data
