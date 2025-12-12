from torch.utils.data import DataLoader

from astrodata.data.schemas import TorchProcessedData, TorchRawData


class TorchDataLoaderWrapper:
    """
    Wrapper around PyTorch DataLoader to integrate with astrodata pipelines.

    This class creates DataLoaders for train/validation/test splits and provides
    common transforms and configurations.
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        Initialize the DataLoader wrapper.

        Args:
            batch_size: Batch size for all dataloaders
            num_workers: Number of subprocesses for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def create_dataloaders(self, raw_data: TorchRawData) -> TorchProcessedData:
        """
        Create PyTorch DataLoaders from TorchRawData.

        Args:
            raw_data: TorchRawData object containing torch datasets

        Returns:
            TorchProcessedData object containing DataLoaders
        """

        datasets = raw_data.data
        dataloaders = {}

        for split, dataset in datasets.items():

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

            dataloaders[split] = dataloader

        processed_metadata = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "original_metadata": raw_data.metadata,
        }

        return TorchProcessedData(dataloaders=dataloaders, metadata=processed_metadata)
