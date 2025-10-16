import torch

from astrodata.data.loaders.torch_dataloader import TorchDataLoaderWrapper
from astrodata.data.loaders.torch_loader import TorchLoader


def test_torch_dataloader_wrapper_with_image(tmp_image_dataset):
    loader = TorchLoader()
    raw = loader.load(str(tmp_image_dataset))

    wrapper = TorchDataLoaderWrapper(batch_size=2, num_workers=0, pin_memory=False)
    processed = wrapper.create_dataloaders(raw)

    assert processed.metadata["batch_size"] == 2
    assert "original_metadata" in processed.metadata
    assert set(processed.dataloaders.keys()).issuperset({"train", "test"})

    train_loader = processed.get_dataloader("train")
    images, labels = next(iter(train_loader))
    assert isinstance(images, torch.Tensor)
    assert images.shape[0] == 2
    assert labels.shape[0] == 2
