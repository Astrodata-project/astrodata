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

---

## Torch vision / FITS data support

Astrodata provides support for PyTorch-like image datasets (classic RGB images and FITS images) through the following components:

### Core Classes

- **`TorchLoader`**  
  High-level loader that:
  * Expects a directory with train[/val]/test splits.
  * Each split contains one subdirectory per class.
  * Automatically infers dataset type (PNG/JPEG vs FITS) by scanning the train split.
  * Returns a `TorchRawData` object containing split-specific `Dataset` instances and metadata.

- **`TorchImageDataset`**  
  Underlying `torch.utils.data.Dataset` for standard image formats (`.png`, `.jpg`, `.jpeg`):
  * Builds `class_to_idx` mapping.
  * Loads images using `torchvision.io.decode_image` (tensor shape `[C, H, W]`).

- **`TorchFITSDataset`**  
  Dataset for FITS images:
  * Assumes 2D single-plane image data (converted to `[1, H, W]`).
  * Uses `astropy.io.fits` to read pixel arrays.

- **`TorchDataLoaderWrapper`**  
  Convenience wrapper that:
  * Consumes a `TorchRawData` object.
  * Builds PyTorch `DataLoader` instances for each split with uniform settings (batch size, workers, pin_memory).
  * Returns a `TorchProcessedData` object.
```{note}:
This wrapper is optional; ML modules can consume `TorchRawData` directly, as they define their own `DataLoader` settings.
```

- **Schemas**
  * `TorchRawData`: Holds split-name → Dataset mapping and metadata (e.g. dataset_type).
  * `TorchProcessedData`: Holds split-name → DataLoader mapping plus loader parameters.

### Expected Directory Structure

```
dataset_root/
    train/
        class_a/
            img1.png | img1.fits
            ...
        class_b/
            ...
    val/            # optional
        class_a/
        class_b/
    test/
        class_a/
        class_b/
```

```{note}:
- `val/` is optional.
- Mixing FITS and PNG/JPEG in the same dataset root is not allowed (the loader will raise).
- Class names are inferred from subdirectory names.
```

### How to Use

Below is a minimal example extracted and simplified from `examples/data/3_torch_data.py`:

```python
from astrodata.data import TorchLoader, TorchDataLoaderWrapper

root = "path/to/dataset_root"

loader = TorchLoader()
raw = loader.load(root)
train_dataset = raw.get_dataset("train")
test_dataset = raw.get_dataset("test")

# Optional
wrapper = TorchDataLoaderWrapper(batch_size=32, num_workers=0, pin_memory=False)
processed = wrapper.create_dataloaders(raw)

train_loader = processed.get_dataloader("train")
for images, labels in train_loader:
    # images: tensor [B, C, H, W]
    # labels: tensor [B]
    break
```

### FITS vs Image Handling

- PNG/JPEG:
  * Decoded via `torchvision.io.decode_image`.
  * No transforms are applied by default; users can wrap datasets or extend the loader for augmentations.
- FITS:
  * Only 2D primary HDU images supported currently.
  * Data coerced to `float32`, single channel with shape `[1, H, W]`.
  * Extend `TorchFITSDataset` to handle multi-extension or multi-channel cases.

### Common Extension Points

- Add transforms to `TorchFITSDataset`: wrap it or modify its `__getitem__`.
- Support more image types: extend valid extensions set.
- Multi-channel FITS: change FITS loading logic (e.g., stack planes).
- Custom sampling strategies: provide `sampler` argument when instantiating `DataLoader`.

Refer to `examples/data/3_torch_data.py` for a full runnable demonstration that prepares CIFAR10-style image directories and a minimal FITS example.

---

## TensorFlow vision / FITS data support

Astrodata also provides a TensorFlow-based loader for image datasets (PNG/JPEG) and FITS files using tf.data and Keras utilities.

### Core Classes

- **`TensorflowLoader`**  
  High-level loader that:
  * Expects a directory with train[/val]/test splits.
  * Each split contains one subdirectory per class.
  * Automatically infers dataset type (PNG/JPEG vs FITS) by scanning the train split.
  * Returns a `TensorflowData` object containing split-specific `tf.data.Dataset` instances and metadata.

- **`TensorflowImageDataset`**  
  Builder around `tf.keras.utils.image_dataset_from_directory`:
  * Supports typical arguments like `image_size`, `batch_size`, `color_mode`, `shuffle`, `seed`, `validation_split`, etc.
  * Propagates `class_names` and `class_to_idx` in metadata.

- **`TensorflowFITSDataset`**  
  FITS dataset builder using `tf.data`:
  * Gathers file paths and labels from class subdirectories.
  * Reads FITS arrays via `tf.numpy_function` wrapping astrodata’s FITS decoder.
  * Returns `(image, label)` pairs, prefetched and optionally batched.

- **Schema**
  * `TensorflowData`: Holds split-name → `tf.data.Dataset` mapping and metadata (dataset_type, class_names, class_to_idx, params).

### Expected Directory Structure

```
dataset_root/
    train/
        class_a/
            img1.png | img1.fits
            ...
        class_b/
            ...
    val/            # optional
        class_a/
        class_b/
    test/
        class_a/
        class_b/
```

```{note}:
- `val/` is optional.
- Mixing FITS and PNG/JPEG in the same dataset root is not allowed (the loader will raise).
- Class names are inferred from subdirectory names.
```

### How to Use

Minimal example:

```python
from astrodata.data import TensorflowLoader

root = "path/to/dataset_root"

loader = TensorflowLoader()
# Forward common Keras directory-loader params for image datasets,
# or batch_size for FITS datasets.
raw = loader.load(
    root,
    image_size=(224, 224),     # for image datasets
    batch_size=32,
    color_mode="rgb",
    shuffle=True,
    seed=42,
)

train_ds = raw.get_dataset("train")
test_ds = raw.get_dataset("test")

for images, labels in train_ds.take(1):
    # images: TensorFlow tensor [B, H, W, C] for image datasets
    #         TensorFlow tensor [B, H, W] or [H, W] for FITS (depending on batching)
    # labels: int tensor
    pass
```

```{warning}:
image_size parameter is mandatory for image datasets.
```

### FITS vs Image Handling

- PNG/JPEG:
  * Uses `tf.keras.utils.image_dataset_from_directory`.
  * Channels-last tensors by default; set `data_format` if needed.
  * Augmentations can be added via `tf.data` map stages or Keras preprocessing layers.

- FITS:
  * Files gathered via class directories; decoded to `float32` tensors using astrodata’s FITS utilities.
  * Built as a `tf.data.Dataset` with `map` and `prefetch`; batching controlled by `batch_size`.
  * Extend `TensorflowFITSDataset` to support multi-extension/multi-channel FITS.

Refer to `examples/data/4_tensorflow_data.py` for a full runnable demonstration that prepares CIFAR10-style image directories and a minimal FITS example.
