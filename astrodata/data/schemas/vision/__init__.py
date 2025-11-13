try:
    from .torch import (
        TorchFITSDataset,
        TorchImageDataset,
        TorchProcessedData,
        TorchRawData,
    )
except Exception:
    HAS_VISION = False
else:
    HAS_VISION = True
