# Extending astrodata

The `astrodata` library is designed with extensibility in mind, leveraging Python's Abstract Base Classes (ABCs) to define clear interfaces for different components. This tutorial will explain what ABCs are, why they are used in `astrodata`, and how you can extend the library by implementing your own custom metrics, models, and model selection strategies.  

## Understanding Abstract Base Classes (ABCs) in Python

An Abstract Base Class (ABC) in Python defines a blueprint for other classes. It allows you to specify methods that *must* be implemented by any concrete (non-abstract) subclass. This enforces a consistent structure and behavior across different implementations of a common concept.

### Why Use ABCs?

1.  **Enforce Interfaces**: ABCs ensure that all subclasses adhere to a specific interface. If a method is marked as abstract in the base class, any concrete subclass *must* provide an implementation for that method, otherwise, it cannot be instantiated.
2.  **Consistency and Predictability**: By defining a common interface, ABCs make your code more predictable. Users of the library know what methods to expect from any class that implements a specific ABC, regardless of its underlying implementation.
3.  **Extensibility**: They provide clear extension points. Developers know exactly what they need to implement to add new functionality (e.g., a new metric or a new model type) while remaining compatible with the rest of the framework.
4.  **Type Hinting and Static Analysis**: ABCs work well with type hinting, allowing for more robust static analysis and clearer code documentation.

Following there will be a series of examples showing how to extend some of the `astrodata` components.

## Example 1, Extending `astrodata.ml.metrics.BaseMetric`

The `BaseMetric` abstract class defines the interface for all evaluation metrics within `astrodata`. If you want to use a custom metric that isn't covered by `SklearnMetric`, you can create your own by inheriting from `BaseMetric`.

**`BaseMetric.py` Abstract Methods:**

  * `__init__(self)`: The constructor.
  * `__call__(self, y_true: Any, y_pred: Any, **kwargs) -> float`: Computes the metric value.
  * `get_name(self) -> str`: Returns the name of the metric.
  * `greater_is_better(self) -> bool` (property): Indicates if a higher value of the metric is better.

**Example: Creating a Custom Precision Metric**

Let's say you want a simple precision metric for binary classification where `greater_is_better` is `True`.

```python
from astrodata.ml.metrics.BaseMetric import BaseMetric
from typing import Any

class CustomPrecisionMetric(BaseMetric):
    def __init__(self, positive_label: Any = 1):
        super().__init__()
        self.positive_label = positive_label

    def __call__(self, y_true: Any, y_pred: Any, **kwargs) -> float:
        true_positives = 0
        predicted_positives = 0
        for true, pred in zip(y_true, y_pred):
            if pred == self.positive_label:
                predicted_positives += 1
                if true == self.positive_label:
                    true_positives += 1
        if predicted_positives == 0:
            return 0.0 # Avoid division by zero
        return true_positives / predicted_positives

    def get_name(self) -> str:
        return f"Precision_for_{self.positive_label}"

    @property
    def greater_is_better(self) -> bool:
        return True

# Example Usage:
y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 1, 0, 0, 1]

precision_metric = CustomPrecisionMetric(positive_label=1)
score = precision_metric(y_true, y_pred)
print(f"Custom Precision Score: {score}")
print(f"Metric Name: {precision_metric.get_name()}")
print(f"Greater is better: {precision_metric.greater_is_better}")
```

## Example 2, Extending `astrodata.ml.models.BaseMlModel`

The `BaseMlModel` abstract class defines the fundamental operations for any machine learning model in `astrodata`. This allows the `astrodata.ml.model_selection` module to work seamlessly with various model types, whether they are scikit-learn models, XGBoost models, or your own custom implementations.

**`BaseMlModel.py` Abstract Methods:**

  * `fit(self, X: Any, y: Any, **kwargs) -> "BaseMlModel"`: Trains the model.
  * `predict(self, X: Any, **kwargs) -> Any`: Generates predictions.
  * `score(self, X: Any, y: Any, **kwargs) -> float`: Computes a default score.
  * `get_metrics(self, X_test: Any, y_test: Any, metrics: List[BaseMetric], **kwargs) -> Dict[str, Any]`: Evaluates multiple metrics.
  * `get_loss_history_metrics(self, X_test: Any, y_test: Any, metrics: List[BaseMetric], **kwargs) -> Dict[str, Any]`: Retrieves metric history during training (optional, but part of the interface).
  * `save(self, filepath: str, **kwargs)`: Saves the model.
  * `load(self, filepath: str, **kwargs) -> "BaseMlModel"`: Loads a model.
  * `get_params(self, **kwargs) -> Dict[str, Any]`: Returns model hyperparameters.
  * `set_params(self, **kwargs) -> None`: Sets model hyperparameters.
  * `clone(self) -> "BaseMlModel"`: Creates a shallow copy.

**Example: Creating a Simple Custom Majority Class Classifier Model**

This example is simplified for illustration purposes. A real custom model would involve more complex machine learning logic.

```python
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.ml.metrics.BaseMetric import BaseMetric
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from sklearn.metrics import accuracy_score
from typing import Any, Dict, List, Optional
import collections
import joblib

class MajorityClassClassifier(BaseMlModel):
    def __init__(self, random_state: int = 42):
        super().__init__()
        self.majority_class = None
        self.random_state = random_state # Although not used for randomness in this simple model, it's good practice for consistency
        self.model_class = self.__class__ # For clone and load methods
        self.model_params = {}

    def fit(self, X: Any, y: Any, **kwargs) -> "MajorityClassClassifier":
        # Find the most frequent class in y
        counts = collections.Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, X: Any, **kwargs) -> Any:
        if self.majority_class is None:
            raise RuntimeError("Model has not been fitted yet.")
        # Predict the majority class for all inputs
        return [self.majority_class] * len(X)

    def score(self, X: Any, y: Any, scorer: Optional[BaseMetric] = None, **kwargs) -> float:
        if self.majority_class is None:
            raise RuntimeError("Model has not been fitted yet.")
        predictions = self.predict(X)
        if scorer is None:
            # Default scorer
            scorer = SklearnMetric(metric=accuracy_score, name="accuracy", greater_is_better=True)
        return scorer(y, predictions)

    def get_metrics(self, X: Any, y: Any, metrics: List[BaseMetric], **kwargs) -> Dict[str, Any]:
        if self.majority_class is None:
            raise RuntimeError("Model has not been fitted yet.")
        predictions = self.predict(X)
        results = {}
        for metric in metrics:
            results[metric.get_name()] = metric(y, predictions)
        return results

    def get_loss_history_metrics(self, X: Any, y: Any, metrics: List[BaseMetric], **kwargs) -> Dict[str, Any]:
        # This simple model does not have a loss history, so we'll raise an error or return empty
        # In a real model, this would track performance over epochs/iterations
        raise AttributeError("MajorityClassClassifier does not support loss history.")

    @property
    def has_loss_history(self) -> bool:
        return False

    def save(self, filepath: str, **kwargs):
        joblib.dump(self.majority_class, filepath)

    def load(self, filepath: str, **kwargs) -> "MajorityClassClassifier":
        self.majority_class = joblib.load(filepath)
        return self

    def get_params(self, **kwargs) -> Dict[str, Any]:
        return {"random_state": self.random_state}

    def set_params(self, **kwargs) -> None:
        if "random_state" in kwargs:
            self.random_state = kwargs["random_state"]

    def clone(self) -> "MajorityClassClassifier":
        new_instance = MajorityClassClassifier(random_state=self.random_state)
        return new_instance

# Example Usage:
X_train = [[1], [2], [3], [4], [5]]
y_train = [0, 0, 1, 0, 1] # Majority class is 0

X_test = [[6], [7]]
y_test = [1, 0]

model = MajorityClassClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Predictions: {predictions}") # Expected: [0, 0]

accuracy = SklearnMetric(metric=accuracy_score)
score = model.score(X_test, y_test, scorer=accuracy)
print(f"Accuracy: {score}") # Expected: 0.5 (one correct, one incorrect)

model.save("majority_classifier.joblib")
loaded_model = MajorityClassClassifier().load("majority_classifier.joblib")
print(f"Loaded model majority class: {loaded_model.majority_class}")
```

## Example 3, Extending `astrodata.ml.model_selection.BaseMlModelSelector`

The `BaseMlModelSelector` abstract class provides the foundation for any model selection strategy (e.g., Grid Search, Hyperparameter Optimization). To create a new model selection algorithm, you would inherit from this class and implement its abstract methods.

**`BaseMlModelSelector.py` Abstract Methods:**

  * `fit(self, X: Any, y: Any, *args, **kwargs) -> "BaseMlModelSelector"`: Runs the model selection process.
  * `get_best_model(self) -> BaseMlModel`: Returns the best model found.
  * `get_best_params(self) -> Dict[str, Any]`: Returns the hyperparameters of the best model.
  * `get_best_metrics(self) -> Dict[str, Any]`: Returns the evaluation metrics for the best model.
  * `get_params(self, **kwargs) -> Dict[str, Any]`: Returns the parameters of the selector itself.

**Conceptual Example: Implementing a Simple Random Search Selector**

Instead of a full code example (which would be quite extensive), let's outline the conceptual approach for a `RandomSearchSelector`:

```python
from astrodata.ml.model_selection.BaseMlModelSelector import BaseMlModelSelector
from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.ml.metrics.BaseMetric import BaseMetric
from typing import Any, Dict, List, Optional
import random

class RandomSearchSelector(BaseMlModelSelector):
    def __init__(
        self,
        model: BaseMlModel,
        param_distributions: dict, # Dictionary with parameter names as keys and distributions/lists as values
        scorer: BaseMetric,
        n_iter: int = 10,
        val_size: float = 0.2,
        random_state: int = 42,
        metrics: Optional[List[BaseMetric]] = None,
        tracker: Any = None, # Placeholder for a tracking object
        log_all_models: bool = False,
    ):
        super().__init__()
        self.model = model
        self.param_distributions = param_distributions
        self.scorer = scorer
        self.n_iter = n_iter
        self.val_size = val_size
        self.random_state = random_state
        self.metrics = metrics if metrics is not None else []
        self.tracker = tracker
        self.log_all_models = log_all_models

        self._best_model = None
        self._best_params = None
        self._best_metrics = None
        self._best_score = float('-inf') if scorer.greater_is_better else float('inf')


    def fit(self, X: Any, y: Any, *args, **kwargs) -> "RandomSearchSelector":
        # Split data into training and validation sets
        # X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(...)

        random.seed(self.random_state)

        for i in range(self.n_iter):
            # 1. Sample hyperparameters randomly from param_distributions
            # sampled_params = self._sample_params(self.param_distributions)

            # 2. Clone the base model and set sampled parameters
            # current_model = self.model.clone()
            # current_model.set_params(**sampled_params)

            # 3. Fit the model on training split and evaluate on validation split
            # current_model.fit(X_train_split, y_train_split)
            # current_score = current_model.score(X_val_split, y_val_split, scorer=self.scorer)
            # current_metrics = current_model.get_metrics(X_val_split, y_val_split, metrics=self.metrics)

            # 4. Log results using tracker if available (similar to HyperOptSelector example)
            # if self.tracker:
            #     with self.tracker.start_run(nested=True, tags={"hp_iteration": i}):
            #         self.tracker.log_params(sampled_params)
            #         self.tracker.log_metrics({self.scorer.get_name(): current_score})
            #         self.tracker.log_metrics(current_metrics)
            #         if self.log_all_models:
            #             self.tracker.log_model(current_model, "model")

            # 5. Update best model if current model is better
            # if (self.scorer.greater_is_better and current_score > self._best_score) or \
            #    (not self.scorer.greater_is_better and current_score < self._best_score):
            #     self._best_score = current_score
            #     self._best_params = sampled_params
            #     self._best_metrics = current_metrics
            #     self._best_model = current_model.clone() # Keep a clone of the best model

        # After all iterations, if a best model was found, fit it on the full training data
        # if self._best_model:
        #     self._best_model.fit(X, y)

        return self

    def get_best_model(self) -> BaseMlModel:
        return self._best_model

    def get_best_params(self) -> Dict[str, Any]:
        return self._best_params

    def get_best_metrics(self) -> Dict[str, Any]:
        return self._best_metrics

    def get_params(self, **kwargs) -> Dict[str, Any]:
        return {
            "model": self.model,
            "param_distributions": self.param_distributions,
            "scorer": self.scorer,
            "n_iter": self.n_iter,
            "val_size": self.val_size,
            "random_state": self.random_state,
            "metrics": self.metrics,
            "tracker": self.tracker,
            "log_all_models": self.log_all_models,
        }

    # Helper method for sampling parameters (would be implemented internally)
    def _sample_params(self, param_distributions: dict) -> dict:
        sampled = {}
        for param, dist in param_distributions.items():
            if isinstance(dist, list):
                sampled[param] = random.choice(dist)
            # Add logic for different distribution types (e.g., uniform, normal)
            # For example: if isinstance(dist, tuple) and dist[0] == 'uniform':
            # sampled[param] = random.uniform(dist[1], dist[2])
            else:
                sampled[param] = dist # If it's a fixed value
        return sampled
```

This conceptual example shows how you would structure a new model selector by implementing the abstract methods and incorporating your specific search logic (in this case, random sampling). The general flow involves iterating through different parameter combinations, training and evaluating models, and keeping track of the best-performing one.
