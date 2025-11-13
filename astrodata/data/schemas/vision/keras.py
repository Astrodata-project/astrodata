from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from astropy.io import fits
from pydantic import BaseModel


class KerasData(BaseModel):

    source: Path | str
    data: Dict[str, tf.data.Dataset]
    metadata: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True

    def get_dataset(self, split: str):
        """
        Get a specific dataset split.

        Args:
            split: The split name ('train', 'val', or 'test')

        Returns:
            The requested dataset

        Raises:
            KeyError: If the split doesn't exist
        """
        if split not in self.data:
            raise KeyError(
                f"Split '{split}' not found. Available splits: {list(self.data.keys())}"
            )
        return self.data[split]


class KerasImageDataset:
    """
    Wrapper around keras.utils.image_dataset_from_directory.
    """

    def __init__(
        self,
        image_dir: str | Path,
        *,
        labels: str | None | List[int] = "inferred",
        label_mode: str | None = "int",
        class_names: Optional[List[str]] = None,
        color_mode: str = "rgb",
        batch_size: Optional[int] = 32,
        image_size: Tuple[int, int] = (256, 256),
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
        Create a tf.data.Dataset using Keras directory loader.
        """
        ds = tf.keras.utils.image_dataset_from_directory(**self.kwargs)
        return ds

    @staticmethod
    def class_to_idx_from_names(class_names: List[str]) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(class_names)}

    def build(self) -> Tuple[tf.data.Dataset, Dict[str, Any]]:
        """
        Returns:
            dataset: The created tf.data.Dataset
            metadata: Dict containing class_names and class_to_idx
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


class KerasFITSDataset:
    """
    FITS dataset builder for folders like:
      root/
        class_a/*.fits
        class_b/*.fits

    Uses tf.data to stream FITS files with minimal defaults.
    """

    def __init__(self, image_dir: str | Path, batch_size: Optional[int] = None):
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size

    def _gather(self) -> Tuple[List[str], List[int], List[str], Dict[str, int]]:
        class_dirs = [d for d in self.image_dir.iterdir() if d.is_dir()]
        class_dirs.sort()
        class_names = [d.name for d in class_dirs]
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        paths = []
        labels = []
        for d in class_dirs:
            files = [p for p in d.iterdir() if p.is_file()]
            files.sort()
            for f in files:
                paths.append(str(f))
                labels.append(class_to_idx[d.name])

        return paths, labels, class_names, class_to_idx

    @staticmethod
    def _read_fits(path_bytes: bytes) -> np.ndarray:
        path = path_bytes.decode("utf-8")
        with fits.open(path) as hdul:
            hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
            if hdu is None or hdu.data is None:
                raise ValueError(f"No image data found in FITS file: {path}")

            data = np.array(hdu.data, dtype=np.float32, copy=True)

            if data.ndim == 2:
                # [H, W] -> [H, W, 1]
                data = np.expand_dims(data, -1)
            elif data.ndim == 3:
                # Accept H,W,C or C,H,W
                if data.shape[0] <= 4:
                    data = np.moveaxis(data, 0, -1)  # C,H,W -> H,W,C
                elif data.shape[-1] <= 4:
                    pass  # already H,W,C
                else:
                    raise ValueError(
                        f"3D FITS data does not look like multi-channel image "
                        f"(shape {data.shape}) in {path}"
                    )
            else:
                raise ValueError(
                    f"Expected 2D or 3D FITS image, got shape {data.shape} in {path}"
                )

            return data  # H, W, C

    def _map_function(self, path, label):
        img = tf.numpy_function(self._read_fits, [path], tf.float32)
        return img, tf.cast(label, tf.int32)

    def build(self) -> Tuple[tf.data.Dataset, Dict[str, Any]]:
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
