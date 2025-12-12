from pathlib import Path
from typing import Any, Dict, Optional

from astrodata.data.loaders.base import BaseLoader
from astrodata.data.schemas.vision.tensorflow import (
    TensorflowData,
    TensorflowFITSDataset,
    TensorflowImageDataset,
)


class TensorflowLoader(BaseLoader):
    """
    TensorFlow data loader for image or FITS datasets organized into
    train/validation/test directory splits.

    Directory structure:
      root/
      ├── train/
      │   ├── class1/
      │   ├── class2/
      │   └── ...
      ├── val/        (optional)
      │   ├── class1/
      │   ├── class2/
      │   └── ...
      └── test/
          ├── class1/
          ├── class2/
          └── ...

    The dataset type is auto-detected by scanning file extensions in the train split.
    Supports image files (.png, .jpg, .jpeg) and FITS files (.fits). Mixed types
    within the same dataset are not allowed.
    """

    def __init__(self) -> None:
        self.dataset_type: Optional[str] = None
        self.dataset_class = None

    def _set_dataset_type(self, dataset_type: str) -> None:
        if dataset_type == "image":
            self.dataset_type = "image"
            self.dataset_class = TensorflowImageDataset
        elif dataset_type == "fits":
            self.dataset_type = "fits"
            self.dataset_class = TensorflowFITSDataset

    def _infer_dataset_type(self, split_dir: Path) -> None:
        """
        Infer dataset type by scanning file extensions under the split directory.
        """
        image_exts = {".png", ".jpg", ".jpeg"}
        fits_ext = ".fits"

        has_image = False
        has_fits = False

        for p in split_dir.rglob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in image_exts:
                has_image = True
            if ext == fits_ext:
                has_fits = True
            if has_image and has_fits:
                raise RuntimeError(
                    "Mixed file types detected. Astrodata currently supports only a specific data format."
                )

        if has_image:
            self._set_dataset_type("image")
        elif has_fits:
            self._set_dataset_type("fits")

    def load(self, path: str, **dataset_kwargs: Any) -> TensorflowData:
        """
        Load TensorFlow datasets from a directory structure with train/test
        (and optional val) splits.

        Args:
            path: Root directory containing the dataset splits.
            **dataset_kwargs: Extra keyword arguments forwarded to the selected
                dataset builder (TensorflowImageDataset or TensorflowFITSDataset),
                e.g., image_size, batch_size, color_mode, shuffle, seed, etc.

        Returns:
            TensorflowData: Object containing built tf.data.Datasets and metadata.

        Raises:
            ValueError: If the root directory does not exist or train/test are missing.
            RuntimeError: If dataset type cannot be inferred or mixed types are found.
        """
        root_path = Path(path)

        if not root_path.exists():
            raise ValueError(f"Root directory {root_path} does not exist")

        train_dir = root_path / "train"
        val_dir = root_path / "val"
        test_dir = root_path / "test"

        if not (train_dir.exists() and test_dir.exists()):
            raise ValueError(f"Expected 'train' and 'test' directories in {root_path}")

        self._infer_dataset_type(train_dir)

        if self.dataset_class is None:
            raise RuntimeError(
                "dataset_class could not be determined. Please make sure that the file types are supported and consistent."
            )

        datasets: Dict[str, Any] = {}

        # Train
        train_builder = self.dataset_class(train_dir, **dataset_kwargs)
        train_ds, train_meta = train_builder.build()
        datasets["train"] = train_ds

        # Test
        test_builder = self.dataset_class(
            test_dir, shuffle=False, **dataset_kwargs
        )  # Comment by Tom: No shuffling for test set otherwise it won't work.
        test_ds, _ = test_builder.build()
        datasets["test"] = test_ds

        # Optional Val
        metadata = {
            "root_path": str(root_path),
            "dataset_type": self.dataset_type,
            "class_names": train_meta.get("class_names", []),
            "class_to_idx": train_meta.get("class_to_idx", {}),
            "params": train_meta.get("params", {}),
        }

        if val_dir.exists():
            val_builder = self.dataset_class(val_dir, **dataset_kwargs)
            val_ds, _ = val_builder.build()
            datasets["val"] = val_ds

        return TensorflowData(source=root_path, data=datasets, metadata=metadata)
