from pathlib import Path

from astrodata.data.loaders.base import BaseLoader
from astrodata.data.processors.base import AbstractProcessor
from astrodata.data.schemas import ProcessedData
from astrodata.data.utils import convert_to_processed_data
from astrodata.utils.utils import read_config


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
        self.project_path = Path(self.config["project_path"]).resolve()
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
        data = self.loader.load(self.project_path / path)
        for processor in self.processor:
            data = processor.process(data)

        processed_data = convert_to_processed_data(data)
        if dump_output:
            processed_data.dump_parquet(
                self.project_path
                / "astrodata_files/processed_data/processed_data.parquet"
            )
        return processed_data
