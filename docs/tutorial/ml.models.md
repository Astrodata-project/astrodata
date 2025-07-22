# models

The `astrodata.ml.models` module provides a unified interface for various machine learning models, abstracting away framework-specific details. It includes a base model class and specialized wrappers for scikit-learn and XGBoost models.

## Core Concepts

  * **`BaseMlModel`**: This abstract base class defines the standard interface for all machine learning models within `astrodata`. Key methods include:
      * `fit(X, y, **kwargs)`: Trains the model.
      * `predict(X, **kwargs)`: Generates predictions.
      * `score(X, y, **kwargs)`: Computes a default score for the model.
      * `get_metrics(X, y, metrics, **kwargs)`: Evaluates a list of specified metrics.
      * `get_loss_history_metrics(X, y, metrics, **kwargs)`: Retrieves metric history during training, if supported.
      * `save(filepath, **kwargs)`: Saves the trained model.
      * `load(filepath, **kwargs)`: Loads a model from a file.
      * `get_params()`: Returns model hyperparameters.
      * `set_params(**kwargs)`: Sets model hyperparameters.
      * `clone()`: Creates a shallow copy of the model.
  * **`SklearnModel`**: This wrapper allows standard scikit-learn models (e.g., `LogisticRegression`, `RandomForestClassifier`) to conform to the `BaseMlModel` interface. It handles initialization, fitting, prediction, and even exposes loss history for models that support it.
  * **`XGBoostModel`**: Similar to `SklearnModel`, this wrapper provides compatibility for XGBoost models (`xgboost.XGBClassifier`, `xgboost.XGBRegressor`). It automatically sets a default `eval_metric` if none is provided and handles XGBoost-specific aspects like loss history tracking.

## How to Use

### Using `SklearnModel`

```python
from sklearn.ensemble import RandomForestClassifier
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate some synthetic data
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate a scikit-learn model wrapped by SklearnModel
sklearn_model = SklearnModel(model_class=RandomForestClassifier, n_estimators=100, random_state=42)

# Fit the model
sklearn_model.fit(X_train, y_train)

# Make predictions
predictions = sklearn_model.predict(X_test)

# Evaluate with a metric
accuracy = SklearnMetric(metric=accuracy_score)
score = sklearn_model.score(X_test, y_test, scorer=accuracy)
print(f"Model score (accuracy): {score}")

# Get other metrics
metrics_to_evaluate = [SklearnMetric(metric=accuracy_score, name="Accuracy")]
evaluated_metrics = sklearn_model.get_metrics(X_test, y_test, metrics=metrics_to_evaluate)
print(f"Evaluated metrics: {evaluated_metrics}")

# Save and load the model
sklearn_model.save("my_rf_model.joblib")
loaded_model = SklearnModel(model_class=RandomForestClassifier) # Model class is needed for loading
loaded_model.load("my_rf_model.joblib")
print("Model loaded successfully!")
```

### Using `XGBoostModel`

```python
from xgboost import XGBClassifier
from astrodata.ml.models.XGBoostModel import XGBoostModel
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate some synthetic data
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate an XGBoost model wrapped by XGBoostModel
xgboost_model = XGBoostModel(model_class=XGBClassifier, n_estimators=50, use_label_encoder=False, eval_metric='logloss', random_state=42)

# Fit the model
xgboost_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)

# Make predictions
predictions = xgboost_model.predict(X_test)

# Evaluate with a metric
logloss_metric = SklearnMetric(metric=log_loss)
score = xgboost_model.score(X_test, y_test, scorer=logloss_metric)
print(f"Model score (log loss): {score}")

# Get loss history metrics
if xgboost_model.has_loss_history:
    loss_history = xgboost_model.get_loss_history_metrics(X_test, y_test, metrics=[logloss_metric])
    print(f"Loss history: {loss_history}")
```