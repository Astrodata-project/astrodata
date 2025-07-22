# model selection

The `astrodata.ml.model_selection` module provides tools for systematically finding the best machine learning model and hyperparameters for a given task. It includes abstract base classes and concrete implementations like Grid Search and Hyperopt-based selection.

## Core Concepts

  * **`BaseMlModelSelector`**: This abstract base class defines the interface for all model selection strategies. Subclasses must implement:
      * `fit(X, y, *args, **kwargs)`: Runs the model selection process on the data.
      * `get_best_model()`: Returns the best `BaseMlModel` instance found.
      * `get_best_params()`: Returns the hyperparameters of the best model.
      * `get_best_metrics()`: Returns the evaluation metrics for the best model.
      * `get_params()`: Returns the parameters of the selector itself.
  * **`GridSearchSelector`**: Implements an exhaustive search over a specified parameter grid. It trains and evaluates models for every combination of hyperparameters, selecting the one that performs best according to a given `scorer`. It supports both single validation split and cross-validation.
  * **`HyperOptSelector`**: Utilizes the `hyperopt` library for efficient hyperparameter optimization. Instead of exhaustive search, `hyperopt` uses Bayesian optimization (Tree-structured Parzen Estimator, TPE) to intelligently explore the parameter space, often finding better results with fewer evaluations compared to traditional grid search. It requires a `param_space` defined using `hyperopt.hp` functions.

## How to Use

### Using `GridSearchSelector`

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.model_selection.GridSearchSelector import GridSearchSelector
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from sklearn.metrics import accuracy_score
import pandas as pd

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the base model
model = SklearnModel(model_class=GradientBoostingClassifier, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
}

# Define the scorer
accuracy_metric = SklearnMetric(metric=accuracy_score, greater_is_better=True)

# Instantiate GridSearchSelector
grid_search = GridSearchSelector(
    model=model,
    param_grid=param_grid,
    scorer=accuracy_metric,
    val_size=0.2, # Use 20% of training data for validation
    random_state=42
)

# Fit the selector
grid_search.fit(X_train, y_train)

# Get the best results
best_params = grid_search.get_best_params()
best_model = grid_search.get_best_model()
best_metrics = grid_search.get_best_metrics()

print(f"Best parameters: {best_params}")
print(f"Best model: {best_model}")
print(f"Best metrics: {best_metrics}")

# You can now use the best_model for further predictions
# predictions = best_model.predict(X_test)
```

#### Using `HyperOptSelector`

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.model_selection.HyperOptSelector import HyperOptSelector
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from sklearn.metrics import accuracy_score
from hyperopt import hp
import pandas as pd

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the base model for HyperOpt (it can be part of the search space)
gradientboost_model = SklearnModel(model_class=GradientBoostingClassifier)

# Define the parameter space using hyperopt.hp functions
param_space = {
    "model": hp.choice("model", [gradientboost_model]), # model can be part of the search space
    "n_estimators": hp.choice("n_estimators", [50, 100]),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
    "max_depth": hp.choice("max_depth", [3, 5]),
}

# Define the scorer
accuracy = SklearnMetric(metric=accuracy_score, name="accuracy", greater_is_better=True)

# Instantiate HyperOptSelector
hos = HyperOptSelector(
    param_space=param_space,
    scorer=accuracy,
    use_cv=False, # Using a single validation split for simplicity
    val_size=0.2,
    random_state=42,
    max_evals=10 # Number of iterations for hyperopt
)

# Fit the selector
hos.fit(X_train, y_train, X_test=X_test, y_test=y_test) # X_test, y_test are for tracking/logging

print("Best parameters found: ", hos.get_best_params())
print("Best metrics: ", hos.get_best_metrics())
```