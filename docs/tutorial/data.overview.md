# Overview

The `astrodata.data` module provides the foundational building blocks for data ingestion and preprocessing in the astrodata ecosystem. It is designed to facilitate the construction of reproducible, modular, and extensible data pipelines for machine learning workflows.

## Overview

At its core, the `astrodata.data` module enables users to:

- **Load data** from various sources and formats (e.g., CSV, Parquet) into a standardized structure.
- **Process data** through a sequence of customizable transformations using processors.
- **Orchestrate data workflows** via the `DataPipeline` class, which manages the flow from raw data loading to processed output.

This modular approach allows users to easily compose, extend, and reuse data workflows, enforcing consistency and reproducibility across projects.

## Modules

- **Loader**: Responsible for reading raw data from disk or other sources and converting it into a `RawData` object (which contains  a pandas DataFrame).
- **Processor**: Implements a transformation or feature engineering step. Each processor inherits from `AbstractProcessor` and defines a `process` method.
- **DataPipeline**: Coordinates the loading and sequential processing of data, producing a final processed dataset ready for downstream tasks.
