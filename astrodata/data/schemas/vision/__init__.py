try:
    from .torch import (
        TorchFITSDataset,
        TorchImageDataset,
        TorchProcessedData,
        TorchRawData,
    )
    from .tensorflow import (
        TensorflowData,
        TensorflowFITSDataset,
        TensorflowImageDataset,
    )
except Exception:
    HAS_VISION = False
else:
    HAS_VISION = True
