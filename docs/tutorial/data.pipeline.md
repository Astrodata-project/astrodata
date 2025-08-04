# Data Pipeline

The `DataPipeline` class in the `astrodata.data` module orchestrates the loading and processing of data through a sequence of processors. It is designed to standardize and streamline data preparation workflows, making it easy to apply a series of transformations to your data.

## Overview

The pipeline consists of three main components:
- **Loader**: Responsible for loading raw data (e.g., from CSV files) into a standardized format.
- **Processors**: A list of processors that sequentially transform the data.

## Example Usage

```python
from astrodata.data import CsvLoader, DataPipeline

# Initialize loader and processors
loader = CsvLoader()
processors = [CustomProcessor()]

# Create the pipeline
pipeline = DataPipeline(
    config_path="example_config.yaml",
    loader=loader,
    processors=processors,
)

# Run the pipeline on your data file
processed_data = pipeline.run("your_data.csv", dump_output=False)
print(processed_data.data.head())
```

