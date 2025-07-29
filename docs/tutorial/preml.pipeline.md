# PremlPipeline

The `PremlPipeline` class in the `astrodata.preml` module orchestrates machine learning preprocessing steps using a configurable sequence of processors. It is designed to prepare your data for supervised ML tasks by handling preconfigured operations like splitting, encoding, and imputing missing values; it also supports custom processors by subclassing the `PremlProcessor` class.

## Overview

The pipeline consists of:
- **Processors**: A list of preprocessing steps (e.g., splitting, encoding, imputing) applied in order.
- **Configuration**: Processors and their parameters can be defined in code or in a YAML config file. Code-defined processors take precedence. Refer to the [Configuration](<project:./configuration.md>) documentation for more details.

One constraint of the `PremlPipeline` is that it requires a `TrainTestSplitter` processor to be included in the pipeline. This processor is essential for splitting the dataset into training and testing sets, which is a common requirement in machine learning workflows.

## Example Usage

```python
from astrodata.data import ProcessedData
from astrodata.preml import OHE, MissingImputator, PremlPipeline, TrainTestSplitter

# Previous steps of the pipeline...
processed_data = ProcessedData(...)

# Define processors
tts = TrainTestSplitter(targets=["target"], test_size=0.2, random_state=42)
ohe = OHE(categorical_columns=["feature2"], numerical_columns=["feature1", "feature3"])
imputer = MissingImputator(categorical_columns=["feature2"], numerical_columns=["feature1", "feature3"])

# Create the pipeline
preml_pipeline = PremlPipeline(
    config_path="example_config.yaml",
    processors=[tts, imputer, ohe],
)

# Run the pipeline
preml_data = preml_pipeline.run(processed_data, dump_output=False)
print(preml_data.train_features.head())
```
