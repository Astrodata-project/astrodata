from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from astropy.io import fits
from torchvision.io import write_png

from astrodata.data.loaders.base import BaseLoader
from astrodata.data.processors.base import AbstractProcessor
from astrodata.data.schemas import RawData


@pytest.fixture(autouse=True)
def set_pandas_options():
    import pandas as pd

    pd.set_option("mode.chained_assignment", None)
    yield
    pd.reset_option("mode.chained_assignment")


class DummyLoader(BaseLoader):
    def load(self, path: str) -> RawData:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        return RawData(source=path, format="csv", data=df)


class DummyProcessor(AbstractProcessor):
    def process(self, raw: RawData) -> RawData:
        # Add a column 'c' as sum of 'a' and 'b'
        df = raw.data.copy()
        df["c"] = df["a"] + df["b"]
        return RawData(source=raw.source, format=raw.format, data=df)


@pytest.fixture
def dummy_config():
    return {
        "project_path": Path("/dummy/project/path").resolve(),
        "preml": {
            "TrainTestSplitter": {"targets": ["tg"], "test_size": 0.2},
            "OHE": {
                "categorical_columns": ["cat"],
                "numerical_columns": ["num"],
            },
        },
    }


@pytest.fixture
def dummy_loader():
    return DummyLoader()


@pytest.fixture
def dummy_processor():
    return DummyProcessor()


@pytest.fixture
def tmp_image_dataset(tmp_path: Path) -> Path:

    root = tmp_path / "image_dataset"
    splits = ["train", "test", "val"]
    classes = ["class1", "class2"]

    for split in splits:
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                img = torch.randint(0, 255, (3, 8, 8), dtype=torch.uint8)
                write_png(img, str(d / f"img_{i}.png"))
    return root


@pytest.fixture
def tmp_fits_dataset(tmp_path: Path) -> Path:

    root = tmp_path / "fits_dataset"
    splits = ["train", "test"]
    classes = ["class1", "class2"]

    for split in splits:
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                data = (np.random.rand(8, 8) * 100).astype("float32")
                fits.writeto(str(d / f"img_{i}.fits"), data, overwrite=True)
    return root
