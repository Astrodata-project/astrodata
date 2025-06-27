from astrodata.data.schemas import ProcessedData
from astrodata.preml.processors import ConvertToPremlData
from astrodata.preml.processors.base import AbstractProcessor
from astrodata.preml.schemas import Premldata
from astrodata.utils.utils import read_config


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
        """
        converter = ConvertToPremlData(self.config)
        data = converter.process(processeddata)
        self.operations_tracker.append(
            {f"{converter.__class__.__name__}": converter.artifact}
        )

        for processor in self.processors:
            if processor.__class__.__name__ in self.config:
                processor.kwargs = self.config[processor.__class__.__name__]
            data = processor.process(data)
            self.operations_tracker.append(
                {f"{processor.__class__.__name__}": processor.artifact}
            )
        print(f"Operations tracker: {self.operations_tracker}")
        return data
