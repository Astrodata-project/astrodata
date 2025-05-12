from astrodata.preml.processors.base import AbstractProcessor
from astrodata.data.schemas import ProcessedData
from astrodata.preml.schemas import Premldata
from astrodata.preml.utils import convert_to_preml_data, read_config


class PremlPipeline:
    """
    A pipeline for processing data using a loader and a series of processors.

    Attributes:
        processors (list[AbstractProcessor]): A list of processors to process the data.
        config_path (str): The path to the configuration file.

    Methods:
        run(path: str) -> Premldata:
            Executes the pipeline by loading a yaml from the given path,
            applying the processors sequentially, and converting the
            result into a Premldata object.
    """

    def __init__(self, processors: list[AbstractProcessor], config_path: str):
        self.processors = processors
        self.operations_tracker = []
        self.config = read_config(config_path).get("preml", {})
        print(f"Config loaded from {config_path}")
        print(f"Config: {self.config}")

    def run(self, processeddata: ProcessedData) -> Premldata:
        """
        Executes the data pipeline.

        Args:
            path (str): The file path to load the raw data from.

        Returns:
            ProcessedData: The processed data after applying all processors.
        """
        data = convert_to_preml_data(processeddata, self.config)

        for processor in self.processors:
            data = processor.process(data)
            self.operations_tracker.append(
                {f"{processor.__class__.__name__}": processor.artifacts}
            )
        return data
