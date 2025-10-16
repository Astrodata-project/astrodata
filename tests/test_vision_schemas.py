import pytest
from torch.utils.data import DataLoader, Dataset

from astrodata.data.schemas.vision import TorchProcessedData, TorchRawData


class _DummyDS(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return idx, 0


def test_torch_rawdata_get_dataset_errors():
    raw = TorchRawData(source=".", data={"train": _DummyDS()}, metadata={})
    with pytest.raises(KeyError):
        raw.get_dataset("invalid")


def test_torch_processeddata_get_dataloader_errors():
    dl = DataLoader(_DummyDS(), batch_size=1)
    processed = TorchProcessedData(dataloaders={"train": dl}, metadata={})
    with pytest.raises(KeyError):
        processed.get_dataloader("invalid")
