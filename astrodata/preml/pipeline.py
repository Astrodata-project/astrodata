from astrodata.data.schemas import ProcessedData
from astrodata.preml.processors import TrainTestSplitter
from astrodata.preml.processors.base import PremlProcessor
from astrodata.preml.schemas import Premldata
from astrodata.utils.utils import instantiate_processors, read_config


class PremlPipeline:
    """
    Pipeline for processing data using a configurable sequence of processors.

    Features:
        - Requires either a config_path or a processors list (not both None).
        - Merges processors from config and argument, with argument processors taking priority.
        - Ensures the first processor is a TrainTestSplitter.

    Args:
        config_path (str, optional): Path to the pipeline configuration file.
        processors (list[PremlProcessor], optional): List of processor instances.

    Methods:
        run(processeddata: ProcessedData) -> Premldata:
            Executes the pipeline, applying processors in order and returning the final Premldata.
    """

    def __init__(
        self, config_path: str = None, processors: list[PremlProcessor] = None
    ):
        if not config_path and not processors:
            raise ValueError("Either config_path or processors must be provided.")
        config = read_config(config_path) if config_path else {}
        self.config = config.get("preml", {}) if config else {}
        self.processors = []
        processors = processors or []
        config_processors = instantiate_processors(self.config) if self.config else []

        # Add TrainTestSplitter as the first processor if not already present
        for p in processors:
            if p.__class__.__name__ == "TrainTestSplitter":
                self.processors.append(p)
                processors.remove(p)
        if "TrainTestSplitter" in config_processors and self.processors == []:
            self.processors.append(config_processors["TrainTestSplitter"])
            config_processors.pop("TrainTestSplitter")
        if not self.processors:
            raise ValueError("A TrainTestSplitter must be defined.")

        # Add the remaining processors, giving priority to argument processors
        for p in processors:
            self.processors.append(p)
        for p in config_processors.values():
            self.processors.append(p)

        self.operations_tracker = []

    def run(self, processeddata: ProcessedData) -> Premldata:
        """
        Executes the data pipeline.
        """
        converter = self.processors[0]
        data = converter.process(processeddata)
        self.operations_tracker.append(
            {f"{converter.__class__.__name__}": converter.artifact}
        )

        for processor in self.processors[1:]:
            if processor.__class__.__name__ in self.config:
                processor.kwargs = self.config[processor.__class__.__name__]
            data = processor.process(data)
            self.operations_tracker.append(
                {f"{processor.__class__.__name__}": processor.artifact}
            )
        return data
