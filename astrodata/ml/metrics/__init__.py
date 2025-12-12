from .BaseMetric import BaseMetric
from .SklearnMetric import SklearnMetric

try:
    from .TensorflowMetric import TensorflowMetric
except (ImportError, ModuleNotFoundError):
    TensorflowMetric = None

def _lazy_import_tensorflow():
    try:
        from .TensorflowMetric import TensorflowMetric
        return TensorflowMetric
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "TensorflowMetric requires TensorFlow dependencies. "
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

TensorflowMetric = _LazyLoader(_lazy_import_tensorflow)