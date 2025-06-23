import pandas as pd
import pytest

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
def dummy_loader():
    return DummyLoader()


@pytest.fixture
def dummy_processor():
    return DummyProcessor()
