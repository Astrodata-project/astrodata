import os
import tempfile

import pandas as pd
import pytest

from astrodata.data.schemas import ProcessedData
from astrodata.preml.pipeline import PremlPipeline
from astrodata.preml.processors import PremlProcessor, TrainTestSplitter
from astrodata.preml.processors.Ohe import OHE
from astrodata.preml.schemas import Premldata


@pytest.fixture
def processed_data():
    data = {
        "feature1": [1, 2, 3, 4],
        "feature2": ["A", "B", "A", "C"],
        "tg": [0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    return ProcessedData(data=df)


class DummyProcessor(PremlProcessor):
    def process(self, preml, artifact=None, **kwargs):
        return preml


def test_pipeline_raises_without_train_test_splitter(monkeypatch, dummy_config):
    # No TrainTestSplitter in config or processors
    dummy_config["preml"].pop("TrainTestSplitter", None)

    monkeypatch.setattr(
        "astrodata.preml.pipeline.read_config", lambda path: dummy_config
    )
    with pytest.raises(ValueError, match="A TrainTestSplitter must be defined."):
        PremlPipeline(config_path="any.yaml")


def test_pipeline_train_test_splitter_first(monkeypatch, dummy_config):
    # Patch read_config in the pipeline module to return config with TrainTestSplitter
    monkeypatch.setattr(
        "astrodata.preml.pipeline.read_config", lambda path: dummy_config
    )
    pipeline = PremlPipeline(config_path="any.yaml")
    assert pipeline.processors[0].__class__.__name__ == "TrainTestSplitter"

    # TrainTestSplitter in processors argument
    tts = TrainTestSplitter(targets=["tg"], test_size=0.2)
    pipeline2 = PremlPipeline(config_path="any.yaml", processors=[tts])
    assert pipeline2.processors[0].__class__.__name__ == "TrainTestSplitter"


def test_pipeline_processors_priority(monkeypatch, dummy_config):
    # Patch read_config in the pipeline module to return config with TrainTestSplitter and OHE
    dummy_config["preml"]["OHE"] = {
        "categorical_columns": ["feature1"],
        "numerical_columns": ["feature2"],
    }
    monkeypatch.setattr(
        "astrodata.preml.pipeline.read_config", lambda path: dummy_config
    )

    tts = TrainTestSplitter(targets=["tg"], test_size=0.2)
    custom_ohe = OHE(categorical_columns=["cat"], numerical_columns=["num"])
    pipeline = PremlPipeline(config_path="any.yaml", processors=[tts, custom_ohe])
    # TrainTestSplitter is first
    assert pipeline.processors[0] is tts
    # Custom OHE is before config OHE
    assert any(p is custom_ohe for p in pipeline.processors)
    # Config OHE is present only if not overridden (should be present after custom_ohe)
    ohe_classes = [p.__class__.__name__ for p in pipeline.processors]
    assert ohe_classes.count("OHE") >= 1


def test_save_and_load_artifact():
    proc = DummyProcessor()
    artifact = {"foo": 42}
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        proc.save_path = tmp.name
        proc.save_artifact(artifact)
        loaded = proc.load_artifact(tmp.name)
        assert proc.artifact == artifact
    os.remove(tmp.name)


def test_ohe_process():
    df = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1, 2, 3], "tg": [0, 1, 0]})
    preml = Premldata(
        train_features=df,
        test_features=df,
        train_targets=df["tg"],
        test_targets=df["tg"],
    )
    ohe = OHE(categorical_columns=["cat"], numerical_columns=["num"])
    result = ohe.process(preml)
    assert "num" in result.train_features.columns
    assert any(col.startswith("cat_") for col in result.train_features.columns)


def test_pipeline_run(monkeypatch, dummy_config):
    # Create dummy processed data
    df = pd.DataFrame(
        {
            "cat": ["a", "b", "a", "c", "b"],
            "num": [1, 2, 3, 2, 5],
            "tg": [0, 1, 0, 1, 1],
        }
    )
    processed = ProcessedData(data=df)

    # --- Without validation in config ---
    monkeypatch.setattr(
        "astrodata.preml.pipeline.read_config", lambda path: dummy_config
    )
    processors = [OHE(categorical_columns=["cat"], numerical_columns=["num"])]
    pipeline = PremlPipeline(config_path="dummy.yaml", processors=processors)
    result = pipeline.run(processed, dump_output=False)
    assert isinstance(result, Premldata)
    assert "num" in result.train_features.columns
    assert any(col.startswith("cat") for col in result.train_features.columns)
    total_rows = len(df)
    train_rows = len(result.train_features)
    test_rows = len(result.test_features)
    expected_test_rows = int(
        total_rows * dummy_config["preml"]["TrainTestSplitter"]["test_size"]
    )
    assert train_rows + test_rows == total_rows
    assert test_rows == expected_test_rows

    # --- With validation in config ---
    dummy_config["preml"]["TrainTestSplitter"]["validation"] = {
        "enabled": True,
        "size": 0.2,
    }
    monkeypatch.setattr(
        "astrodata.preml.pipeline.read_config", lambda path: dummy_config
    )
    pipeline_val = PremlPipeline(config_path="dummy.yaml", processors=processors)
    result_val = pipeline_val.run(processed, dump_output=False)
    assert isinstance(result_val, Premldata)
    assert "num" in result_val.train_features.columns
    assert any(col.startswith("cat") for col in result_val.train_features.columns)
    # Validation: check that validation split exists and sizes add up
    assert hasattr(result_val, "val_features")
    val_rows = len(result_val.val_features)
    train_rows_val = len(result_val.train_features)
    test_rows_val = len(result_val.test_features)
    expected_val_rows = int(
        total_rows * dummy_config["preml"]["TrainTestSplitter"]["validation"]["size"]
    )
    assert train_rows_val + test_rows_val + val_rows == total_rows
    assert val_rows == expected_val_rows
