# Getting Started

To start using astrodata you can install it as any python package through the GitHub link

## Installation

Install the package directly from GitHub using any package manager such as pip, uv, and conda, the package requires `python >= 3.10`.
```sh
pip install git+https://github.com/Astrodata-project/astrodata.git

conda install git+https://github.com/Astrodata-project/astrodata.git

uv add git+https://github.com/Astrodata-project/astrodata.git
```

## Basic Usage

astrodata at its base works as any other python package, by importing the desired modules and working with the intended workflows astrodata facilitates building machine learning pipelines from data import to model training, implementing tracking and reproducibility along the way.

The [astrotaxi](<project:./python_examples/astrotaxi/0_astrotaxi_example.rst>) example outlines a full pipeline from data import to model training.

Taking the data import part of the example:
```python
from astrodata.data import AbstractProcessor, DataPipeline, ParquetLoader, RawData

loader = ParquetLoader()

class TargetCreator(AbstractProcessor):
    def process(self, raw: RawData) -> RawData:
        raw.data["duration"] = (
            raw.data.lpep_dropoff_datetime - raw.data.lpep_pickup_datetime
        )
        raw.data["duration"] = raw.data["duration"].apply(
            lambda x: x.total_seconds() / 60
        )
        raw.data = raw.data[
            (raw.data["duration"] >= 1) & (raw.data["duration"] <= 60)
        ].reset_index(drop=True)
        raw.data = raw.data[raw.data["trip_distance"] < 50].reset_index(drop=True)
        return raw

data_processors = [TargetCreator()]

data_pipeline = DataPipeline(
    config_path=config, loader=loader, processors=data_processors
)

data_path = "./testdata/green_tripdata_2024-01.parquet"

processed = data_pipeline.run(data_path)
tracker.track("Data pipeline run, processed data versioned")

```

You can see that we start by importing the required classes `AbstractProcessor, DataPipeline, ParquetLoader, RawData` and then by performing operations using the functions that said classes contain. The package is made so that each element can work independently but at the same time respects a "common" pipeline of data -> preml -> ml, with tracking being present along all steps in different forms.

## Dependencies

As per the `pyproject.toml`:

```
dependencies = [
    "dvc>=3.59.2",
    "gitpython>=3.1.44",
    "hyperopt>=0.2.7",
    "mlflow>=2.22.0",
    "numpy>=2.2.4",
    "pandas>=2.2.3",
    "pyarrow>=19.0.1",
    "pydantic>=2.11.3",
    "scikit-learn>=1.6.1",
    "tqdm>=4.67.1",
    "xgboost>=3.0.0",
]
```

`tensorflow` and `pytorch` are optional dependencies in case either of them is required (e.g. for their respective `PytorchModel` and `TensorflowModel`).

## FAQ

TODO: to be collected once researchers start using the package.

## Links to further documentation

Following is the documentation of some of the included packages:
- [**pandas**](https://pandas.pydata.org/docs/)
- [**scikit-learn**](https://scikit-learn.org/stable/)
- [**xgboost**](https://xgboost.readthedocs.io/en/stable/)
- [**hyperopt**](https://hyperopt.github.io/hyperopt/)
- [**mlflow**](https://mlflow.org/docs/latest/ml/)
  
## Contribution Guidelines

Contribution is handled through GitHub pull requests, new functions that correctly extend the provided abstract classes can be accepted without too much control, refer to [extending astrodata](<project:./python_examples/tutorial/extending.md>) for further informations.