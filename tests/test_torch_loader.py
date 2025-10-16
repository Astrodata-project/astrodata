import pytest
import torch

from astrodata.data.loaders.torch_loader import TorchLoader


def test_torch_loader_image_dataset(tmp_image_dataset):
    loader = TorchLoader()
    raw = loader.load(str(tmp_image_dataset))

    assert raw.metadata["dataset_type"] == "image"
    assert set(raw.data.keys()).issuperset({"train", "test"})
    assert "val" in raw.data

    train_ds = raw.get_dataset("train")
    x, y = train_ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.dim() == 3  # C,H,W
    assert isinstance(y, int)
    assert len(train_ds) > 0


def test_torch_loader_fits_dataset(tmp_fits_dataset):
    loader = TorchLoader()
    raw = loader.load(str(tmp_fits_dataset))

    assert raw.metadata["dataset_type"] == "fits"
    assert set(raw.data.keys()) == {"train", "test"}

    train_ds = raw.get_dataset("train")
    x, y = train_ds[0]
    assert isinstance(x, torch.Tensor)
    # For 2D FITS, dataset returns [1, H, W]
    assert x.dim() == 3 and x.shape[0] in (1, 3, 4)
    assert isinstance(y, int)
    assert len(train_ds) > 0
