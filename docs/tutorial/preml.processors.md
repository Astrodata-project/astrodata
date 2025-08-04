# processors

The `astrodata.preml.processors` module provides a framework to perform advanced preprocessing and transformation steps on your data after it has been loaded and split into training, testing, and (optionally) validation sets. Similar to the `astrodata.data.processors` module, it is built around an extensible interface that allows users to compose preprocessing workflows chaining multiple operations together.

## Abstract Class

The core of the `astrodata.preml.processors` module is the `PremlProcessor` abstract base class, which defines the interface for all preml processors. Each processor is responsible for transforming a `Premldata` object, and can optionally save or load artifacts (such as fitted encoders or imputers) to ensure reproducibility and consistency across different runs.

## How to Use

### Creating a Premlprocessor

```python
from astrodata.preml import Premldata, PremlProcessor

class CustomProcessor(PremlProcessor):
    def process(self, preml: Premldata) -> Premldata:
        # Transform the input Premldata and return the result.
```

- Subclasses must implement the `process` method, which takes a `Premldata` object and returns a transformed `Premldata`.
- Artifacts (such as fitted parameters) can be saved in the process method with `PremlProcessor.save_artifact()`, and a processor can be initialized with an existing artifact. 
- Processors can be chained together in a pipeline for complex preprocessing workflows.

### Using built-in Processors

The `astrodata.preml.processors` module provides several built-in processors for common preprocessing tasks:

#### TrainTestSplitter

Splits your dataset into training, testing, and optionally validation sets. You can specify target columns, test size, random state, and validation split. The output is a `Premldata` object containing the split datasets and metadata.

```python
from astrodata.preml import TrainTestSplitter

splitter = TrainTestSplitter(
    targets=["target_column"],
    test_size=0.2,
    random_state=42,
    validation={"enabled": True, "size": 0.1},
)
```

#### OHE (OneHotEncoder)

One-hot encodes specified categorical columns, optionally retaining numerical columns. Artifacts can be saved for reproducibility.

```python
from astrodata.preml import OHE

encoder = OHE(
    categorical_columns=["cat1", "cat2"],
    numerical_columns=["num1", "num2"],
)
```

#### MissingImputator

Imputes missing values in numerical columns (using the mean) and categorical columns (using the most frequent value).

```python
from astrodata.preml import MissingImputator

imputer = MissingImputator(
    numerical_columns=["num1", "num2"],
    categorical_columns=["cat1", "cat2"],
)
```

#### Standardizer

Standardizes numerical columns to have zero mean and unit variance. Useful for scaling features before machine learning.

```python
from astrodata.preml import Standardizer

scaler = Standardizer(
    numerical_columns=["num1", "num2"],
)
```
