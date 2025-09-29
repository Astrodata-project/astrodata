# processors

The `astrodata.data.processors` module provides a framework for transforming data within the astrodata workflow. It is built around an extensible interface that allows users to compose preprocessing workflows by chaining multiple operations together. This module is designed to be flexible, enabling users to create custom processors that can be applied to their data in a modular way.

## Abstract Class

**`AbstractProcessor`** is the abstract base class for all processors. Subclasses must implement:
  * `process(raw: RawData) -> RawData`: Applies a transformation to the input `RawData` and returns the result.

This interface ensures that all processors can be chained together in a pipeline, regardless of their specific function.

## How to Use

### Creating a Processor

To define a custom processor, subclass `AbstractProcessor` and implement the `process` method:

```python
from astrodata.data import RawData, AbstractProcessor

class FeatureAdder(AbstractProcessor):
    def process(self, raw: RawData) -> RawData:
        raw.data["feature_sum"] = raw.data["feature1"] + raw.data["feature2"]
        return raw
```

### Using Built-in Processors

The `astrodata.data.processors.common` module includes some simple ready-to-use processors to showcase the functionality. These processors can be used directly in your data pipelines:

- **`NormalizeAndSplit`**: Normalizes data by subtracting the mean and dividing by the standard deviation.
- **`DropDuplicates`**: Removes duplicate rows from the dataset.

Example usage:

```python
from astrodata.data import NormalizeAndSplit, DropDuplicates

processors = [NormalizeAndSplit(), DropDuplicates(), FeatureAdder()]
```

## Extensibility

To add new preprocessing steps, simply create a new processor by subclassing `AbstractProcessor`. Processors can be combined in any order, allowing for flexible and reusable data transformations.

```{hint}
Processors are applied in sequence by the `DataPipeline`, order matters! Ensure that the sequence of processors is logical for your data transformation needs.
```
