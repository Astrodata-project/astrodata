from .BaseMlModel import BaseMlModel
from .SklearnModel import SklearnModel
from .XGBoostModel import XGBoostModel

def _lazy_import_pytorch():
    try:
        from .PytorchModel import PytorchModel
        return PytorchModel
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "PytorchModel requires PyTorch dependencies. "
            "Install with: uv sync --extra torch"
        ) from e

def _lazy_import_tensorflow():
    try:
        from .TensorflowModel import TensorflowModel
        return TensorflowModel
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "TensorflowModel requires TensorFlow dependencies. "
            "Install with: uv sync --extra tensorflow"
        ) from e

class _LazyLoader:
    def __init__(self, import_func):
        self._import_func = import_func
        self._module = None
    
    def __call__(self, *args, **kwargs):
        if self._module is None:
            self._module = self._import_func()
        return self._module(*args, **kwargs)
    
    def __getattr__(self, name):
        if self._module is None:
            self._module = self._import_func()
        return getattr(self._module, name)

PytorchModel = _LazyLoader(_lazy_import_pytorch)
TensorflowModel = _LazyLoader(_lazy_import_tensorflow)