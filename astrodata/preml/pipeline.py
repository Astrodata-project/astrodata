from pathlib import Path

from astrodata.data.schemas import ProcessedData
from astrodata.preml.processors.base import PremlProcessor
from astrodata.preml.schemas import Premldata
from astrodata.preml.utils import instantiate_processors
from astrodata.utils.utils import read_config


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

    def __init__(self, config_path: str, processors: list[PremlProcessor] = None):

        self.config = read_config(config_path)
        self.project_path = Path(self.config["project_path"]).resolve()
        self.preml_config = self.config.get("preml", {})
        if not self.config and not processors:
            raise ValueError(
                "Either preml section of config or processors must be provided."
            )
        self.processors = []
        processors = processors or []
        config_processors = instantiate_processors(self.preml_config)

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
            if p.__class__.__name__ not in [
                processor.__class__.__name__ for processor in self.processors
            ]:
                self.processors.append(p)

        self.operations_tracker = []

    def run(self, processeddata: ProcessedData, dump_output: bool = True) -> Premldata:
        """
        Executes the data pipeline.
        """
        converter = self.processors[0]
        data = converter.process(processeddata)
        self.operations_tracker.append(
            {f"{converter.__class__.__name__}": converter.artifact}
        )
        if dump_output:
            preml_path = self.project_path / "astrodata_files/preml"
            preml_path.mkdir(parents=True, exist_ok=True)

        for processor in self.processors[1:]:
            if dump_output:
                processor.save_path = (
                    self.project_path
                    / f"astrodata_files/preml/{processor.__class__.__name__}.pkl"
                )
            if processor.__class__.__name__ in self.preml_config:
                processor.kwargs = self.preml_config[processor.__class__.__name__]
            data = processor.process(data)
            self.operations_tracker.append(
                {f"{processor.__class__.__name__}": processor.artifact}
            )
        if dump_output:
            data.dump_parquet(self.project_path / "astrodata_files/preml/")
        return data
