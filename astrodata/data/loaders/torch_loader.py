from pathlib import Path

from astrodata.data.loaders.base import BaseLoader
from astrodata.data.schemas import TorchImageDataset, TorchRawData


class TorchLoader(BaseLoader):
    """
    PyTorch data loader for image datasets with train/validation/test splits.

    This loader expects a directory structure like:
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
    """

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
            raise ValueError(f"Expected train/test directories in {root_path}")

        datasets = {}

        datasets["train"] = TorchImageDataset(train_dir)
        datasets["test"] = TorchImageDataset(test_dir)

        if val_dir.exists():
            datasets["val"] = TorchImageDataset(val_dir)

        metadata = {
            "data_type": "torch_image_dataset",
            "root_path": str(root_path),
        }

        return TorchRawData(
            source=root_path, format="torch_image", data=datasets, metadata=metadata
        )
