from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, ConfigDict

from astrodata.data.utils import FITS_EXTS, decode_fits, gather_paths_and_labels


class TensorflowData(BaseModel):

    source: Path | str
    data: Dict[str, tf.data.Dataset]
    metadata: Dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_dataset(self, split: str):
        """
        Return a specific dataset split.

        Args:
            split: Split name, typically "train", "val", or "test".

        Returns:
            tf.data.Dataset for the requested split.

        Raises:
            KeyError: If the split is not present in self.data.
        """
        if split not in self.data:
            raise KeyError(
                f"Split '{split}' not found. Available splits: {list(self.data.keys())}"
            )
        return self.data[split]


class TensorflowImageDataset:
    """
    Wrapper around tf.keras.utils.image_dataset_from_directory with convenience
    parameters for common image classification setups. Supports optional
    validation splits and propagates class names/indices in metadata.
    """

    def __init__(
        self,
        image_dir: str | Path,
        image_size: Tuple[int, int],
        *,
        labels: str | None | List[int] = "inferred",
        label_mode: str | None = "int",
        class_names: Optional[List[str]] = None,
        color_mode: str = "rgb",
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        validation_split: Optional[float] = None,
        subset: Optional[
            str
        ] = None,  # "training", "validation" when using validation_split
        interpolation: str = "bilinear",
        follow_links: bool = False,
        crop_to_aspect_ratio: bool = False,
        pad_to_aspect_ratio: bool = False,
        data_format: Optional[str] = None,  # "channels_last" | "channels_first"
        verbose: bool = True,
    ):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise ValueError(f"Directory {self.image_dir} does not exist")

        self.kwargs = dict(
            directory=str(self.image_dir),
            labels=labels,
            label_mode=label_mode,
            class_names=class_names,
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            seed=seed,
            validation_split=validation_split,
            subset=subset,
            interpolation=interpolation,
            follow_links=follow_links,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            data_format=data_format,
            verbose=verbose,
        )

    def _build(self) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset using Tensorflow directory loader.
        """
        ds = tf.keras.utils.image_dataset_from_directory(**self.kwargs)
        return ds

    @staticmethod
    def class_to_idx_from_names(class_names: List[str]) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(class_names)}

    def build(self) -> Tuple[tf.data.Dataset, Dict[str, Any]]:
        """
        Build the dataset and collect metadata.

        Returns:
            dataset: The created tf.data.Dataset.
            metadata: Dict containing:
              - class_names: Ordered list of class names.
              - class_to_idx: Mapping from class name to integer index.
              - image_dir: Source directory.
              - params: Loader parameters (excluding directory).
        """
        ds = self._build()
        names = getattr(ds, "class_names", []) or []
        meta = {
            "class_names": names,
            "class_to_idx": self.class_to_idx_from_names(names) if names else {},
            "image_dir": str(self.image_dir),
            "params": {k: v for k, v in self.kwargs.items() if k != "directory"},
        }
        return ds, meta


class TensorflowFITSDataset:
    """
    TensorFlow FITS dataset builder for directory trees of the form:
      root/
        class_a/*.fits
        class_b/*.fits

    Streams FITS images via tf.data and numpy_function with minimal defaults,
    returning (image, label) pairs and class metadata.
    """

    def __init__(self, image_dir: str | Path, batch_size: Optional[int] = None):
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size

    def _gather(self) -> Tuple[List[str], List[int], List[str], Dict[str, int]]:
        paths, labels, class_names, class_to_idx = gather_paths_and_labels(
            self.image_dir, valid_exts=FITS_EXTS, return_type="str"
        )
        return paths, labels, class_names, class_to_idx

    @staticmethod
    def _read_fits(path_bytes: bytes) -> np.ndarray:
        path = path_bytes.decode("utf-8")
        return decode_fits(path)

    def _map_function(self, path, label):
        img = tf.numpy_function(self._read_fits, [path], tf.float32)
        return img, tf.cast(label, tf.int32)

    def build(self) -> Tuple[tf.data.Dataset, Dict[str, Any]]:
        """
        Build a FITS tf.data.Dataset and collect metadata.

        Returns:
            dataset: Prefetched (and optionally batched) tf.data.Dataset of (image, label).
            metadata: Dict containing:
              - class_names: Ordered list of class names.
              - class_to_idx: Mapping from class name to integer index.
              - image_dir: Source directory.
        """
        paths, labels, class_names, class_to_idx = self._gather()

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(self._map_function, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        if self.batch_size is not None:
            ds = ds.batch(self.batch_size)

        meta = {
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "image_dir": str(self.image_dir),
        }
        return ds, meta
