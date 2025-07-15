from astrodata.data.pipeline import DataPipeline
from astrodata.data.processors.base import AbstractProcessor
from astrodata.data.schemas import ProcessedData, RawData


def test_pipeline_runs_and_processes(monkeypatch, dummy_loader, dummy_processor):
    config = {"project_path": "."}
    monkeypatch.setattr("astrodata.data.pipeline.read_config", lambda path: config)
    pipeline = DataPipeline(
        config_path="dummy.yaml", loader=dummy_loader, processors=[dummy_processor]
    )
    processed = pipeline.run("dummy_path.csv")
    assert isinstance(processed, ProcessedData)
    assert "c" in processed.data.columns
    assert all(processed.data["c"] == processed.data["a"] + processed.data["b"])


def test_pipeline_multiple_processors(monkeypatch, dummy_loader):
    class AddOneProcessor(AbstractProcessor):
        def process(self, raw: RawData) -> RawData:
            df = raw.data.copy()
            df["a"] = df["a"] + 1
            return RawData(source=raw.source, format=raw.format, data=df)

    config = {"project_path": "."}
    monkeypatch.setattr("astrodata.data.pipeline.read_config", lambda path: config)
    pipeline = DataPipeline(
        config_path="dummy.yaml",
        loader=dummy_loader,
        processors=[AddOneProcessor(), AddOneProcessor()],
    )
    processed = pipeline.run("dummy_path.csv")
    assert all(processed.data["a"] == [3, 4])


def test_pipeline_empty_processors(monkeypatch, dummy_loader):
    config = {"project_path": "."}
    monkeypatch.setattr("astrodata.data.pipeline.read_config", lambda path: config)
    pipeline = DataPipeline(
        config_path="dummy.yaml", loader=dummy_loader, processors=[]
    )
    processed = pipeline.run("dummy_path.csv")
    assert isinstance(processed, ProcessedData)
    assert list(processed.data.columns) == ["a", "b"]
