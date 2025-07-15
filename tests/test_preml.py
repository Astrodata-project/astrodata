import os
import tempfile

import pandas as pd
import pytest

from astrodata.data.schemas import ProcessedData
from astrodata.preml.pipeline import PremlPipeline
from astrodata.preml.processors import PremlProcessor, TrainTestSplitter
from astrodata.preml.processors.Ohe import OHE
from astrodata.preml.schemas import Premldata

CONFIG_WITH_VALIDATION = """
preml:
  TrainTestSplitter:
    targets:
      - "tg"
    test_size: 0.2
    random_state: 42
    validation:
      enabled: true
      size: 0.1
"""

CONFIG_NO_VALIDATION = """
preml:
  TrainTestSplitter:
    targets:
      - "tg"
    test_size: 0.2
    random_state: 42
    validation:
      enabled: false
      size: 0.1
"""

CONFIG_ALL_PROCS = """
preml:
  TrainTestSplitter:
    targets:
      - "tg"
    test_size: 0.2
    random_state: 42
    validation:
      enabled: false
      size: 0.1
  OHE:
    categorical_columns:
      - "feature2"
    numerical_columns:
      - "feature1"
  MissingImputator:
    categorical_columns:
      - "feature2"
    numerical_columns:
      - "feature1"
"""


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


def test_pipeline_raises_without_train_test_splitter(monkeypatch):
    # No TrainTestSplitter in config or processors
    config = {
        "project_path": ".",
        "preml": {
            "OHE": {"categorical_columns": ["cat"], "numerical_columns": ["num"]}
        },
    }
    monkeypatch.setattr("astrodata.preml.pipeline.read_config", lambda path: config)
    with pytest.raises(ValueError, match="A TrainTestSplitter must be defined."):
        PremlPipeline(config_path="any.yaml")


def test_pipeline_train_test_splitter_first(monkeypatch):
    # Patch read_config in the pipeline module to return config with TrainTestSplitter
    config = {
        "project_path": ".",
        "preml": {"TrainTestSplitter": {"targets": ["tg"], "test_size": 0.2}},
    }
    monkeypatch.setattr("astrodata.preml.pipeline.read_config", lambda path: config)
    pipeline = PremlPipeline(config_path="any.yaml")
    assert pipeline.processors[0].__class__.__name__ == "TrainTestSplitter"

    # TrainTestSplitter in processors argument
    tts = TrainTestSplitter(targets=["tg"], test_size=0.2)
    pipeline2 = PremlPipeline(config_path="any.yaml", processors=[tts])
    assert pipeline2.processors[0].__class__.__name__ == "TrainTestSplitter"


def test_pipeline_processors_priority(monkeypatch):
    # Patch read_config in the pipeline module to return config with TrainTestSplitter and OHE
    config = {
        "project_path": ".",
        "preml": {
            "TrainTestSplitter": {"targets": ["tg"], "test_size": 0.2},
            "OHE": {
                "categorical_columns": ["feature2"],
                "numerical_columns": ["feature1"],
            },
        },
    }
    monkeypatch.setattr("astrodata.preml.pipeline.read_config", lambda path: config)

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


def test_pipeline_run(monkeypatch):
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
    config_no_val = {
        "project_path": ".",
        "preml": {
            "TrainTestSplitter": {"targets": ["tg"], "test_size": 0.2},
            "OHE": {
                "categorical_columns": ["cat"],
                "numerical_columns": ["num"],
            },
        },
    }
    monkeypatch.setattr(
        "astrodata.preml.pipeline.read_config", lambda path: config_no_val
    )
    processors = [OHE(categorical_columns=["cat"], numerical_columns=["num"])]
    pipeline = PremlPipeline(config_path="dummy.yaml", processors=processors)
    result = pipeline.run(processed)
    assert isinstance(result, Premldata)
    assert "num" in result.train_features.columns
    assert any(col.startswith("cat") for col in result.train_features.columns)
    total_rows = len(df)
    train_rows = len(result.train_features)
    test_rows = len(result.test_features)
    expected_test_rows = int(
        total_rows * config_no_val["preml"]["TrainTestSplitter"]["test_size"]
    )
    assert train_rows + test_rows == total_rows
    assert test_rows == expected_test_rows

    # --- With validation in config ---
    config_with_val = {
        "project_path": ".",
        "preml": {
            "TrainTestSplitter": {
                "targets": ["tg"],
                "test_size": 0.2,
                "validation": {"enabled": True, "size": 0.2},
            },
            "OHE": {
                "categorical_columns": ["cat"],
                "numerical_columns": ["num"],
            },
        },
    }
    monkeypatch.setattr(
        "astrodata.preml.pipeline.read_config", lambda path: config_with_val
    )
    pipeline_val = PremlPipeline(config_path="dummy.yaml", processors=processors)
    result_val = pipeline_val.run(processed)
    assert isinstance(result_val, Premldata)
    assert "num" in result_val.train_features.columns
    assert any(col.startswith("cat") for col in result_val.train_features.columns)
    # Validation: check that validation split exists and sizes add up
    assert hasattr(result_val, "val_features")
    val_rows = len(result_val.val_features)
    train_rows_val = len(result_val.train_features)
    test_rows_val = len(result_val.test_features)
    expected_val_rows = int(
        total_rows * config_with_val["preml"]["TrainTestSplitter"]["validation"]["size"]
    )
    assert train_rows_val + test_rows_val + val_rows == total_rows
    assert val_rows == expected_val_rows
