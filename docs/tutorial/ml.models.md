# models

The `astrodata.ml.models` module provides a unified interface for various machine learning models, abstracting away framework-specific details. It includes a base model class and specialized wrappers for the most common model frameworks.

## Abstract Class

**`BaseMlModel`** is the abstract base class of any model and it defines the standard interface for all machine learning models within `astrodata`. Subclasses must implement:
  * `fit(X, y, **kwargs)`: Trains the model on training data.
  * `predict(X, **kwargs)`: Generates predictions for input data.
  * `score(X, y, **kwargs)`: Computes a default score for the model.
  * `get_scorer_metric()`: Returns the default metric used by the score method.
  * `get_metrics(X, y, metrics, **kwargs)`: Evaluates a list of specified metrics.
  * `save(filepath, **kwargs)`: Saves the trained model to disk.
  * `load(filepath, **kwargs)`: Loads a model from a file.
  * `clone()`: Creates a deep copy of the model instance.

Subclasses may optionally implement:
  * `get_params()`: Returns model hyperparameters (raises `NotImplementedError` by default).
  * `set_params(**kwargs)`: Sets model hyperparameters (raises `NotImplementedError` by default).
  * `get_loss_history()`: Retrieves training loss history, if supported by the model.
  * `get_loss_history_metrics(X, y, metrics, **kwargs)`: Computes metrics at each training stage, if supported.
  * `has_loss_history` (property): Boolean indicating if loss history is available.

## How to Use

### Initializing a Model

How a model is initialized depends on the framework of reference. In general, the goal should be to pass an existing model of the chosen framework and then let the class handle the generalization to the `astrodata` framework.

```python
from sklearn.svm import LinearSVR
from astrodata.ml.models.SklearnModel import SklearnModel

model = SklearnModel(model_class=LinearSVR, random_state=42)
```

Looking at this example which initializes a `scikit-learn` model we can see that the `model_class` is passed along with any extra class specific argument (in this case `random_state=42`) to initialize the model.

```{hint}
Model initialization is the only *framework-specific* part of a model, all other methods are framework-agnostic, examples will be shown for a generic `BaseMlModel`.
```

### Fitting a model

Once a model is initialized correctly, fitting it requires an `X_train` and a `y_train` to be passed through its `fit` method. The internal logic of the model should handle the rest of the training.

```python
model.fit(X_train, y_train)
```

This will result in a set of weights to be computed for the model, which will be later used for predicting new values.


### Predicting with a fitted model

A model that has been correctly fitted can be used for predictions by invoking its `predict` method.

```python
y_pred = model.predict(x_test)
```

The output of the predict method is an array containing the predicted labels for the given input.

### Computing metrics

Given an array of [`BaseMetric`](<project:./ml.metrics.md>) that has been previously created, we can compute the metrics for our model by invoking its `get_metrics` method, this will output a dictionary with the computed values for each metric.

```python
metrics = model.get_metrics(
        X_test,
        y_test,
        metrics=metrics,
    )
```

## `SklearnModel`

A wrapper for [**scikit-learn**](https://scikit-learn.org/stable/) models that provides a standardized interface consistent with `astrodata`. It can be initialized using any scikit-learn estimator class.

### Key Features
- **Unified interface**: Standard `fit`, `predict`, `score`, and `get_metrics` methods
- **Loss history tracking**: Available for models that support staged predictions (e.g., `GradientBoostingClassifier`)
- **Automatic scorer detection**: Automatically selects appropriate default metrics based on model type (classifier, regressor, clusterer, outlier detector)
- **Serialization**: Save and load models using joblib
- **Parameter management**: `get_params()`, `set_params()`, and `clone()` support

### Example
```python
from sklearn.ensemble import RandomForestClassifier
from astrodata.ml.models import SklearnModel

model = SklearnModel(
    model_class=RandomForestClassifier,
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## `XGBoostModel`

A wrapper for [**XGBoost**](https://xgboost.readthedocs.io/en/stable/) models that provides a standardized interface consistent with `astrodata`. It supports both XGBoost classifiers and regressors.

### Key Features
- **Unified interface**: Standard `fit`, `predict`, `score`, and `get_metrics` methods
- **Loss history tracking**: Access training loss history via `get_loss_history()` and staged metrics via `get_loss_history_metrics()`
- **Automatic eval_metric**: Sets default evaluation metric based on model type if not specified
- **Serialization**: Save and load models using joblib
- **Parameter management**: `get_params()`, `set_params()`, and `clone()` support

### Example
```python
import xgboost as xgb
from astrodata.ml.models import XGBoostModel

model = XGBoostModel(
    model_class=xgb.XGBClassifier,
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## `PytorchModel`

A lightweight wrapper around [PyTorch](https://pytorch.org/) modules that provides a unified training and evaluation interface consistent with the rest of `astrodata`. It accepts either an instantiated `nn.Module` or a class with `model_params`, and exposes convenience helpers for metric computation, training history tracking, saving/loading, validation monitoring, and layer freezing for fine-tuning.

### Key Features
- **Flexible initialization**: Accept instantiated models or model classes with parameters
- **Automatic device management**: Supports CPU and CUDA with automatic detection
- **Training history tracking**: Per-step training metrics and per-epoch validation metrics
- **Multiple save formats**: Supports PyTorch native (`.pt`), pickle (`.pkl`), and SafeTensors formats
- **Fine-tuning support**: Freeze and unfreeze specific layers or all layers
- **Dataset flexibility**: Works with NumPy arrays, PyTorch tensors, or custom `DataLoader` instances
- **Parameter management**: `get_params()`, `set_params()`, and `clone()` support

### Initializing

```python
import torch.nn.functional as F
from torch import nn, optim
from astrodata.ml.models import PytorchModel

class SimpleClassifier(nn.Module):
    def __init__(self, input_layers, output_layers):
        super().__init__()
        self.fc1 = nn.Linear(input_layers, 64)
        self.fc2 = nn.Linear(64, output_layers)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = PytorchModel(
    model_class=SimpleClassifier,
    model_params={"input_layers": X_train.shape[1], "output_layers": n_classes},
    loss_fn=nn.CrossEntropyLoss,
    optimizer=optim.AdamW,
    optimizer_params={"lr": 1e-3},
    epochs=10,
    batch_size=32,
    device="cpu",  # or "cuda" if available
)
```

### Training

Train from arrays/tensors:

```python
model.fit(X=X_train, y=y_train)
```

Alternatively, use a custom `Dataset` or `DataLoader` for full control over batching and transforms:

```python
from torch.utils.data import Dataset, TensorDataset

# Option 1: Pass a Dataset
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
model.fit(dataset=train_ds, epochs=10, batch_size=32)

# Option 2: The model will create a DataLoader internally from arrays
model.fit(X=X_train, y=y_train, epochs=10, batch_size=32)
```

### Validation metrics (per-epoch)

Optionally pass validation data and metrics to track validation performance at the end of each epoch. Training metrics are recorded per batch during the epoch.

```python
from sklearn.metrics import accuracy_score
from astrodata.ml.metrics import SklearnMetric

metrics = [SklearnMetric(accuracy_score)]

model.fit(
    X=X_train,
    y=y_train,
    X_val=X_val,
    y_val=y_val,
    metrics=metrics,
    epochs=10,
    batch_size=32,
)

# Retrieve histories
train_history = model.get_metrics_history(split="train")  # per-batch training metrics
val_history = model.get_metrics_history(split="val")      # per-epoch validation metrics
```

### Fine-tuning with frozen layers

For transfer learning, you can selectively freeze layers before fine-tuning:

```python
# Freeze all layers
model.freeze_layers("all")

# Unfreeze specific layers for fine-tuning
model.unfreeze_layers(["fc2"])

# Fine-tune with existing weights
model.fit(X=X_train, y=y_train, fine_tune=True, epochs=5)
```

### Predicting

```python
y_pred = model.predict(X_test, batch_size=32)
y_proba = model.predict_proba(X_test, batch_size=32)
```

### Computing metrics

```python
scores = model.get_metrics(X=X_test, y=y_test, metrics=metrics, batch_size=32)
```

### Saving and loading

Multiple formats are supported for serialization:

```python
# PyTorch native format (recommended)
model.save("model.pt", format="torch")
model.load("model.pt", format="torch")

# Pickle format
model.save("model.pkl", format="pkl")
model.load("model.pkl", format="pkl")

# SafeTensors format (for model weights only)
model.save("model.safetensors", format="safetensors")
model.load("model.safetensors", format="safetensors")
```

### Periodic checkpointing during training

Save model checkpoints at regular intervals during training:

```python
model.fit(
    X=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    save_every_n_epochs=10,
    save_folder="checkpoints/",
    save_format="torch"
)
# This will save checkpoint_10.pt, checkpoint_20.pt, etc.
```

## `TensorflowModel`

A lightweight wrapper around TensorFlow/Keras models providing a unified training and prediction interface consistent with the rest of `astrodata`. It accepts either an instantiated `keras.Model` or a model class with `model_params`, and exposes convenience helpers for metric computation, training history tracking, saving/loading, validation monitoring, and layer freezing for fine-tuning.

### Key Features
- **Flexible initialization**: Accept instantiated Keras models or model classes with parameters
- **Automatic device management**: TensorFlow handles device placement automatically
- **Training history tracking**: Built-in Keras history tracking for training and validation metrics
- **Multiple save formats**: Supports Keras native (`.keras`), HDF5 (`.h5`), and SavedModel formats
- **Fine-tuning support**: Freeze and unfreeze specific layers or all layers, with support for nested models
- **Dataset flexibility**: Works with NumPy arrays, TensorFlow tensors, or `tf.data.Dataset` instances
- **Parameter management**: `get_params()`, `set_params()`, and `clone()` support

### Initializing

```python
import keras as K
from astrodata.ml.models import TensorflowModel

class SimpleClassifier(K.Model):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = K.layers.Dense(64, activation='relu')
        self.fc2 = K.layers.Dense(output_dim, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)

model = TensorflowModel(
    model_class=SimpleClassifier,
    model_params={"input_dim": X_train.shape[1], "output_dim": n_classes},
    loss_fn=K.losses.SparseCategoricalCrossentropy,
    optimizer=K.optimizers.Adam,
    optimizer_params={"learning_rate": 1e-3},
    epochs=10,
    batch_size=32,
    device=None,  # TensorFlow handles device automatically
)
```

### Training

Train from arrays/tensors:

```python
model.fit(X=X_train, y=y_train)
```

Alternatively, use a custom `tf.data.Dataset` for full control over batching and transforms:

```python
import tensorflow as tf

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000).batch(32)
model.fit(dataset=train_ds)
```

### Validation metrics (per-epoch)

Optionally pass validation data and Keras metrics to track validation performance:

```python
import keras as K

metrics = [K.metrics.SparseCategoricalAccuracy(name="accuracy")]

model.fit(
    X=X_train,
    y=y_train,
    X_val=X_val,
    y_val=y_val,
    metrics=metrics,
    epochs=10,
    batch_size=32,
)

# Retrieve histories (Keras native format)
train_history = model.get_metrics_history(split="train")
val_history = model.get_metrics_history(split="val")
```

### Fine-tuning with frozen layers

For transfer learning, you can selectively freeze layers before fine-tuning:

```python
# Freeze all layers
model.freeze_layers("all")

# Unfreeze specific layers for fine-tuning
model.unfreeze_layers(["dense_1"])

# Fine-tune with existing weights
model.fit(X=X_train, y=y_train, fine_tune=True, epochs=5)

# For nested models, specify parent layer
model.freeze_layers(["conv_block"], parent_layer="feature_extractor")
```

### Predicting

```python
y_pred = model.predict(X_test, batch_size=32)
y_proba = model.predict_proba(X_test, batch_size=32)
```

### Computing metrics

TensorflowModel works with both scikit-learn metrics and custom metrics:

```python
from sklearn.metrics import accuracy_score
from astrodata.ml.metrics import SklearnMetric

metrics = [SklearnMetric(accuracy_score)]
scores = model.get_metrics(X=X_test, y=y_test, metrics=metrics, batch_size=32)

# Or with a tf.data.Dataset
scores = model.get_metrics(dataset=test_dataset, metrics=metrics)
```

### Saving and loading

Multiple formats are supported for serialization:

```python
# Keras native format (recommended, TensorFlow 2.x+)
model.save("model.keras", format="tensorflow")
model.load("model.keras", format="tensorflow")

# HDF5 format (legacy compatibility)
model.save("model.h5", format="h5")
model.load("model.h5", format="h5")

# SavedModel format (for TensorFlow Serving)
model.save("saved_model/", format="savedmodel")
model.load("saved_model/", format="savedmodel")
```

### Periodic checkpointing during training

Save model checkpoints at regular intervals during training:

```python
model.fit(
    X=X_train,
    y=y_train,
    epochs=100,
    batch_size=32,
    save_every_n_epochs=10,
    save_folder="checkpoints/",
    save_format="tensorflow"
)
# This will save model_epoch_10.keras, model_epoch_20.keras, etc.
```

```{note}
TensorflowModel works seamlessly with both scikit-learn metrics (via `SklearnMetric`) and Keras native metrics for evaluation.
```

## Examples

### SklearnModel Examples
- [Basic `SkLearnModel` usage](<project:../python_examples/ml/1_sklearn_example.rst>)
- [Multi `SkLearnModel` model training through `for` loops](<project:../python_examples/ml/2_multimodel_example.rst>)

### GridSearch Examples
- [Basic GridSearch usage](<project:../python_examples/ml/3_gridsearch_example.rst>)
- [Parallel GridSearch example](<project:../python_examples/ml/3_1_gridsearch_parallel_example.rst>)
- [Parallel GridSearch comparison](<project:../python_examples/ml/3_2_gridsearch_parallel_comparison.rst>)

### HyperOpt Examples
- [Basic HyperOpt usage](<project:../python_examples/ml/4_hyperopt_example.rst>)
- [Parallel HyperOpt example](<project:../python_examples/ml/4_1_hyperopt_parallel_example.rst>)

### MLflow Examples
- [Simple MLflow example](<project:../python_examples/ml/5_mlflow_simple_example.rst>)
- [MLflow + GridSearch](<project:../python_examples/ml/6_mlflow_gs_example.rst>)
- [MLflow + HyperOpt](<project:../python_examples/ml/7_mlflow_hp_example.rst>)

### PytorchModel Examples
- [Basic `PytorchModel` usage](<project:../python_examples/ml/8_pytorch_example.rst>)
- [`PytorchModel` + GridSearch](<project:../python_examples/ml/9_pytorch_gs_example.rst>)
- [`PytorchModel` + HyperOpt](<project:../python_examples/ml/10_pytorch_hp_example.rst>)
- [`PytorchModel` + MLflow](<project:../python_examples/ml/11_pytorch_mlflow_example.rst>)
- [`PytorchModel` freeze and fine-tune](<project:../python_examples/ml/12_pytorch_freeze_train.rst>)
- [`PytorchModel` with ResNet18](<project:../python_examples/ml/13_pytorch_resnet18.rst>)
- [`PytorchModel` save and load](<project:../python_examples/ml/14_pytorch_save_example.rst>)

### TensorflowModel Examples
- [Basic `TensorflowModel` usage](<project:../python_examples/ml/15_tensorflow_example.rst>)
- [`TensorflowModel` + GridSearch](<project:../python_examples/ml/16_tensorflow_gs_example.rst>)
- [`TensorflowModel` + HyperOpt](<project:../python_examples/ml/17_tensorflow_hp_example.rst>)
- [`TensorflowModel` + MLflow](<project:../python_examples/ml/18_tensorflow_mlflow_example.rst>)
- [`TensorflowModel` freeze and fine-tune](<project:../python_examples/ml/19_tensorflow_freeze_train.rst>)
