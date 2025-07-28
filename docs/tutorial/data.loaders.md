# loaders

The `astrodata.data.loaders` submodule provides a flexible and extensible framework for loading data from various file formats into the astrodata pipeline. It includes a base loader class and specialized loaders for the most common data formats in data science and astrophysics.

## Abstract Class

**`BaseLoader`** is the abstract base class for all data loaders. Subclasses must implement:
  * `load(path)`: Loads data from the specified path and returns a `RawData` object.

This standard interface ensures that all loaders can be used interchangeably within the astrodata pipeline.

## How to Use

### Initializing a Loader

To load data, initialize the loader corresponding to your file format. For example, to load a CSV file:

```python
from astrodata.data import CsvLoader

loader = CsvLoader()
```

### Loading Data

Once initialized, use the loader’s `load` method to read data from disk:

```python
raw_data = loader.load("data.csv")
```

The returned `RawData` object wraps the loaded data (typically within a pandas DataFrame) and is ready for further processing.

## Available Loaders

- **`CsvLoader`**: Loads data from CSV files.
- **`ParquetLoader`**: Loads data from Parquet files.

## Extensibility

The loaders module is designed for easy extension. To add support for a new format (e.g., HDF5, FITS), subclass `BaseLoader` and implement the `load` method for your format. Planned future releases will include loaders for standard astrophysics formats.
