from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image

from astrodata.data.utils import VALID_IMAGE_EXTS, decode_fits, gather_paths_and_labels


class TorchRawData(BaseModel):
    """
    Represents raw PyTorch datasets loaded from directory trees.

    Expects train/validation/test splits under a root, with class subfolders
    containing image files (PNG/JPG/JPEG) or FITS files, depending on dataset type.

    Attributes:
        source: Root directory containing the datasets.
        data: Mapping of split name to torch.utils.data.Dataset (e.g., train/val/test).
        metadata: Information about classes, splits, and class index mapping.
    """

    source: Path | str
    data: Dict[str, Dataset]
    metadata: Dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    Holds DataLoaders per split (e.g., train/val/test) and training-related metadata
    such as batch size, shuffling, num_workers, and applied transforms.
    """

    dataloaders: Dict[str, DataLoader]  # Dictionary of DataLoader objects
    metadata: Dict[str, Any]

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    Loads standard image files and returns tensors suitable for models. Images are
    expected in class-labeled subdirectories. Optional transform is applied to the
    decoded image tensor.
    """

    def __init__(
        self,
        image_dir: str,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.

        Args:
            image_dir: Directory containing images organized by class folders
            transform: Optional transform to apply to each image
        """
        self.image_dir = Path(image_dir)

        # Collect all image files and their labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.transform = transform

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
            idx: Index of the sample to retrieve.

        Returns:
            (image_tensor, label):
              - image_tensor: torch.Tensor with shape [C, H, W] and dtype uint8 by
                default from torchvision.io.decode_image. If a transform is provided,
                its output shape/dtype may differ (e.g., float32).
              - label: int class index.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = decode_image(str(img_path))

        if self.transform:
            image = self.transform(image)

        return image, label


class TorchFITSDataset(Dataset):
    """
    Custom PyTorch Dataset for FITS image data with train/validation/test splits.

    Expects FITS images organized in class subfolders under a split directory.
    Decodes FITS to float32 arrays and returns CHW tensors. Optional transform
    can be applied to the tensor.
    """

    def __init__(self, image_dir: str, transform: Optional[Callable] = None):
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
            (image_tensor, label):
              - image_tensor: torch.Tensor with shape [C, H, W], dtype inferred from
                decode_fits (typically float32). If a transform is provided, its output
                may change shape/dtype.
              - label: int class index.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read FITS image as H, W, C float32 and convert to CHW tensor
        data_hwc = decode_fits(str(img_path))
        data_chw = np.moveaxis(data_hwc, -1, 0)  # H,W,C -> C,H,W
        tensor = torch.from_numpy(data_chw)

        return tensor, label
