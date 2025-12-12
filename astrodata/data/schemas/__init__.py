from .core import ProcessedData, RawData

try:
    from .vision import (
        TorchFITSDataset,
        TorchImageDataset,
        TorchProcessedData,
        TorchRawData,
    )
except Exception:
    HAS_VISION = False
else:
    HAS_VISION = True
