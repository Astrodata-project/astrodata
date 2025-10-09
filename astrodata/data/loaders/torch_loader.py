from pathlib import Path

from astrodata.data.loaders.base import BaseLoader
from astrodata.data.schemas import TorchFITSDataset, TorchImageDataset, TorchRawData


class TorchLoader(BaseLoader):
    """
        PyTorch data loader for image datasets with train/validation/test splits.

        Expected directory structure:
        data_root/
        ├── train/
        │   ├── class1/
        │   ├── class2/
        │   └── ...
        ├── val/  --optional
        │   ├── class1/
        │   ├── class2/
        │   └── ...
        └── test/
            ├── class1/
            ├── class2/
            └── ...

    It can auto-detect dataset type by file extensions or you can provide it.
    """

    def __init__(self):
        self.dataset_type = None
        self.dataset_class = None

    def _set_dataset_type(self, dataset_type: str) -> None:
        if dataset_type == "image":
            self.dataset_type = "image"
            self.dataset_class = TorchImageDataset
        elif dataset_type == "fits":
            self.dataset_type = "fits"
            self.dataset_class = TorchFITSDataset

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

    def load(self, path: str) -> TorchRawData:
        """
        Load PyTorch datasets from directory structure.

        Args:
            path: Root directory containing train/val/test folders

        Returns:
            TorchRawData object containing the loaded datasets
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

        datasets = {}
        datasets["train"] = self.dataset_class(train_dir)
        datasets["test"] = self.dataset_class(test_dir)

        if val_dir.exists():
            datasets["val"] = self.dataset_class(val_dir)

        metadata = {
            "root_path": str(root_path),
            "dataset_type": self.dataset_type,
        }

        return TorchRawData(source=root_path, data=datasets, metadata=metadata)
