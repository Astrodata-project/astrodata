from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image


class RawData(BaseModel):
    """
    Represents raw input data loaded from a source file.

    Attributes:
        source (str): The source of the data (e.g., file path or URL).
        format (Literal): The format of the data (e.g., "fits", "hdf5", "csv", "parquet").
        data (pd.DataFrame): The actual data as a Pandas DataFrame.
    """

    source: Path | str
    format: Literal["fits", "hdf5", "csv", "parquet"]
    data: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


class ProcessedData(BaseModel):
    """
    Represents processed data after transformations.

    Attributes:
        data (pd.DataFrame): The actual data as a Pandas DataFrame.
        metadata (Optional[dict]): Additional metadata about the processed data.
    """

    data: pd.DataFrame
    metadata: Optional[dict] = {}

    class Config:
        arbitrary_types_allowed = True

    def dump_parquet(self, path: Path):
        """
        Dumps the processed data to a Parquet file.

        Args:
            path (Path): The file path to save the Parquet file.
        """
        self.data.to_parquet(path, index=False)


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

    # TODO: controllare che la prima dimensione sia il canale

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

        class_dirs = [d for d in self.image_dir.iterdir() if d.is_dir()]
        class_dirs.sort()

        self.class_to_idx = {
            cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)
        }

        valid_extensions = {".jpg", ".jpeg", ".png"}

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

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

        class_dirs = [d for d in self.image_dir.iterdir() if d.is_dir()]
        class_dirs.sort()

        self.class_to_idx = {
            cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)
        }

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            for img_path in class_dir.iterdir():
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

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

        with fits.open(str(img_path)) as hdul:
            data = hdul[0].data

            if data is None:
                raise ValueError(f"No image data found in FITS file: {img_path}")

            if data.ndim == 2:
                data_native = np.asarray(data, dtype=np.float32)
                tensor = torch.from_numpy(data_native).unsqueeze(0)
            else:
                raise ValueError(
                    f"Expected 2D FITS image, got {data.ndim}D in {img_path}"
                )

        return tensor, label
