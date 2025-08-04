# Overview

The `astrodata.preml` submodule is designed to perform advanced preprocessing and transformation steps on your data after it has been loaded and split into training, testing, and (optionally) validation sets. While the initial data pipeline handles loading and basic preprocessing, `preml` focuses on preparing your datasets for machine learning tasks by applying transformations such as one-hot encoding, missing value imputation, and other feature engineering steps.

The core concept of `preml` is to provide a flexible and reproducible way to chain together multiple preprocessing operations using the `PremlPipeline` class. This pipeline ensures that all transformations are applied consistently across your train, test, and validation splits, maintaining data integrity.

Typical usage involves:
- Defining a `TrainTestSplitter` to split your processed data.
- Specifying additional processors (e.g., `OHE` for categorical encoding, `MissingImputator` for handling missing values).
- Running the `PremlPipeline` to apply these transformations in sequence.

This approach allows you to easily configure, extend, and track your preprocessing steps, ensuring your data is ready for downstream machine learning workflows.

