from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image
from astrodata.data.utils import (
    VALID_IMAGE_EXTS,
    gather_paths_and_labels,
    decode_fits,
)


class TorchRawData(BaseModel):
    """
    Represents raw PyTorch datasets loaded from image directories.

    This schema is specifically designed for PyTorch image datasets
    organized in train/validation/test splits with class folders.

    Attributes:
        source: Root directory containing the datasets
        data: Dictionary of PyTorch datasets (train/val/test)
        metadata: Information about classes, splits, etc.
    """

    source: Path | str
    data: Dict[str, Dataset]
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


class TorchProcessedData(BaseModel):
    """
    Represents processed PyTorch data after transformations and DataLoader creation.

    This schema holds DataLoaders and training-related metadata.

    Attributes:
        dataloaders: Dictionary of PyTorch DataLoaders
        metadata: Information about batch size, transforms, etc.
    """

    dataloaders: Dict[str, DataLoader]  # Dictionary of DataLoader objects
    metadata: Dict[str, Any]

    class Config:
        arbitrary_types_allowed = True

    def get_dataloader(self, split: str):
        """
        Get a specific DataLoader split.

        Args:
            split: The split name ('train', 'val', or 'test')

        Returns:
            The requested DataLoader

        Raises:
            KeyError: If the split doesn't exist
        """
        if split not in self.dataloaders:
            raise KeyError(
                f"Split '{split}' not found. Available splits: {list(self.dataloaders.keys())}"
            )
        return self.dataloaders[split]


class TorchImageDataset(Dataset):
    """
    Custom PyTorch Dataset for image data with train/validation/test splits.

    This dataset loads images from specified directories.
    It expects images to be organized in folders by class/label.
    """

    def __init__(
        self,
        image_dir: str,
    ):
        """
        Initialize the dataset.

        Args:
            image_dir: Directory containing images organized by class folders
        """
        self.image_dir = Path(image_dir)

        # Collect all image files and their labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_image_paths()

    def _load_image_paths(self):
        """Load image paths and create class mappings."""
        if not self.image_dir.exists():
            raise ValueError(f"Directory {self.image_dir} does not exist")

        paths, labels, class_names, class_to_idx = gather_paths_and_labels(
            self.image_dir, valid_exts=VALID_IMAGE_EXTS, return_type="path"
        )

        self.image_paths = paths
        self.labels = labels
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = decode_image(str(img_path))

        return image, label


class TorchFITSDataset(Dataset):
    """
    Custom PyTorch Dataset for FITS image data with train/validation/test splits.

    Expects images organized in class folders under a split directory.
    """

    def __init__(self, image_dir: str):
        """
        Initialize the dataset.

        Args:
            image_dir: Directory containing FITS images organized by class folders
        """
        self.image_dir = Path(image_dir)

        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_image_paths()

    def _load_image_paths(self):
        """Load FITS image paths and create class mappings."""
        paths, labels, class_names, class_to_idx = gather_paths_and_labels(
            self.image_dir, valid_exts=None, return_type="path"
        )
        self.image_paths = paths
        self.labels = labels
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image_tensor, label), where image tensor is shape [C, H, W]
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read FITS image as H, W, C float32 and convert to CHW tensor
        data_hwc = decode_fits(str(img_path))
        data_chw = np.moveaxis(data_hwc, -1, 0)  # H,W,C -> C,H,W
        tensor = torch.from_numpy(data_chw)

        return tensor, label
