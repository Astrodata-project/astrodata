# metrics

The `astrodata.ml.metrics` module provides a standardized interface for defining and using evaluation metrics for machine learning models. It features an abstract base class `BaseMetric` and an adapter for scikit-learn metrics, `SklearnMetric`.

## Core Concepts

  * **`BaseMetric`**: This is an abstract base class that all custom metrics should inherit from. It defines the fundamental methods that a metric object must implement:
      * `__init__()`: Initializes the metric.
      * `__call__(y_true, y_pred, **kwargs)`: Computes the metric value given true and predicted labels.
      * `get_name()`: Returns the name of the metric.
      * `greater_is_better`: A property indicating whether a higher value of the metric is desirable.
  * **`SklearnMetric`**: This class acts as an adapter, allowing you to use any scikit-learn-compatible metric function within the `astrodata` framework. It wraps a callable metric function (e.g., `sklearn.metrics.accuracy_score`) and exposes it through the `BaseMetric` interface.

## How to Use

### Creating a Custom Metric (inheriting from `BaseMetric`)

While not explicitly shown in the provided files, a custom metric would typically look like this:

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

#### Using `SklearnMetric`

You can easily wrap existing scikit-learn metric functions:

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