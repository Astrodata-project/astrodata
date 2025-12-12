# metrics

The `astrodata.ml.metrics` module provides a standardized interface for defining and using evaluation metrics for machine learning models. It features an abstract base class `BaseMetric` and an adapter for scikit-learn metrics, `SklearnMetric`.

## Abstract Class

**`BaseMetric`** is the abstract base class that all custom metrics should inherit from. A metric must implement:
* `__init__()`: Initializes the metric.
* `__call__(y_true, y_pred, **kwargs)`: Computes the metric value given true and predicted labels.
* `get_name()`: Returns the name of the metric.
* `greater_is_better`: A property indicating whether a higher value of the metric is desirable.

## How to Use

### Calling a metric

A metric that has been created following the defined abstract class can be always called directly:

```python
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from sklearn.metrics import accuracy_score

accuracy_metric = SklearnMetric(accuracy_score)

accuracy_computed = accuracy_metric(y_true, y_pred)
```

The result of this operation is the computed metric on the two provided arrays.

```{attention}
`y_true` and `y_pred` should always have the same length by definition.
```
```{tip}
Some sk_learn metrics, especially for classification tasks, require the probability of a given class rather than the predicted label. Be sure to read the related documentation before using them!
```

### Creating a Custom Metric (inheriting from `BaseMetric`)

A custom metric would typically look like this:

```python
from astrodata.ml.metrics.BaseMetric import BaseMetric
from typing import Any

class MyCustomAccuracy(BaseMetric):
    def __init__(self):
        pass

    def __call__(self, y_true: Any, y_pred: Any, **kwargs) -> float:
        # Implement your custom accuracy calculation here
        correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct_predictions / len(y_true)

    def get_name(self) -> str:
        return "MyCustomAccuracy"

    @property
    def greater_is_better(self) -> bool:
        return True
```

### `SklearnMetric`

This class allows you to easily wrap existing scikit-learn metric functions:

```python
from sklearn.metrics import accuracy_score, f1_score
from astrodata.ml.metrics.SklearnMetric import SklearnMetric

# Wrap accuracy_score
accuracy = SklearnMetric(metric=accuracy_score, name="accuracy", greater_is_better=True)

# Wrap f1_score with specific parameters
f1_macro = SklearnMetric(metric=f1_score, name="f1_macro", greater_is_better=True, average='macro')

# Example usage
y_true = [0, 1, 0, 1]
y_pred = [0, 0, 0, 1]

print(f"Accuracy: {accuracy(y_true, y_pred)}")
print(f"F1 Macro: {f1_macro(y_true, y_pred)}")
```

### `TensorflowMetric`

This class allows you to wrap Keras/TensorFlow metric objects while maintaining compatibility with both the Keras metrics interface and the astrodata `BaseMetric` interface:

```python
from keras.metrics import Accuracy, FBetaScore
from astrodata.ml.metrics.TensorflowMetric import TensorflowMetric
import numpy as np

# Wrap a Keras Accuracy metric
accuracy = TensorflowMetric(metric=Accuracy(), name="accuracy", greater_is_better=True)

# Wrap a Keras FBetaScore metric with specific parameters
f1_score = TensorflowMetric(metric=FBetaScore(beta=1.0), name="f1_score", greater_is_better=True)

# Example usage with tensor-like data
y_true = np.array([0, 1, 0, 1])
y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]])

print(f"Accuracy: {accuracy(y_true, y_pred)}")
print(f"F1 Score: {f1_score(y_true, y_pred)}")

# Can also be used as a standard Keras metric
accuracy.update_state(y_true, y_pred)
result = accuracy.result()
print(f"Keras-style result: {result.numpy()}")
accuracy.reset_state()
```

```{note}
`TensorflowMetric` inherits from both `keras.metrics.Metric` and `BaseMetric`, allowing it to be used seamlessly in Keras workflows while providing the astrodata metrics interface.
```

```{tip}
When using TensorFlow/Keras metrics, the input data will be automatically converted to tensors. Make sure your data is in a format compatible with TensorFlow operations.
```