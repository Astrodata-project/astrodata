# models

The `astrodata.ml.models` module provides a unified interface for various machine learning models, abstracting away framework-specific details. It includes a base model class and specialized wrappers for the most common model frameworks.

## Abstract Class

**`BaseMlModel`** is the abstract base class of any model and it defines the standard interface for all machine learning models within `astrodata`. Subclasses must implement:
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

The output of the preidct method is an array containing the predicted lables for the given input.

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

Can be initialized using any of the models offered by the [**scikit-learn**](https://scikit-learn.org/stable/) library, refer to the library documentation for more information on available models and model-specific hyperparameters.

## `XgboostModel`

Can be initialized using any of the models offered by the [**xgboost**](https://xgboost.readthedocs.io/en/stable/) library, refer to the library documentation for more information on available models and model-specific hyperparameters.

## `PytorchModel`

A thin wrapper around [PyTorch](https://pytorch.org/) modules that provides a unified training and evaluation interface consistent with the rest of `astrodata`. It accepts either an instantiated `nn.Module` or a class with `model_params`, and exposes convenience helpers for metric computation, basic history tracking, saving/loading, and optional validation monitoring.

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

Alternatively, use a custom `DataLoader` for full control over batching and transforms:

```python
from torch.utils.data import DataLoader, TensorDataset

train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
model.fit(dataloader=train_loader)
```

### Validation metrics (per-epoch)

Optionally pass validation data and metrics to track validation performance at the end of each epoch. Training-step metrics are also recorded during the epoch.

```python
from sklearn.metrics import accuracy_score, f1_score, log_loss
from astrodata.ml.metrics import SklearnMetric

metrics = [
    SklearnMetric(accuracy_score, greater_is_better=True),
    SklearnMetric(f1_score, average="micro"),
    SklearnMetric(log_loss),
]

model.fit(
    X=X_train,
    y=y_train,
    X_val=X_val,
    y_val=y_val,
    metrics=metrics,
)

# Retrieve histories
train_history = model.get_metrics_history()              # per-step within epoch
val_history = model.get_metrics_history(split="val")     # per-epoch
```

### Predicting

```python
y_pred = model.predict(X_test, batch_size=32)
y_proba = model.predict_proba(X_test, batch_size=32)
```

### Computing metrics

```python
scores = model.get_metrics(X_test, y_test, metrics=metrics)
```

### Freezing layers (fine-tuning)

```python
model.freeze_layers(["fc2"])  # unfreezes only selected module names
```

### Saving and loading

```python
model.save("model.ckpt", format="torch")          # or "pkl" or "safetensors"
model.load("model.ckpt", format="torch")
```

Refer to the examples below for end-to-end usage patterns including MLflow tracking and search.

## `TensorflowModel`

```{attention}
To be implemented in future releases.
```

## Examples

- [Basic `SkLearnModel` usage](<project:../python_examples/ml/1_sklearn_example.rst>)
- [Multi `SkLearnModel` model training through `for` loops](<project:../python_examples/ml/2_multimodel_example.rst>)
- [Basic `PytorchModel` usage](<project:../python_examples/ml/8_pytorch_example.rst>)
- [`PytorchModel` + GridSearch](<project:../python_examples/ml/9_pytorch_gs_example.rst>)
- [`PytorchModel` + HyperOpt](<project:../python_examples/ml/10_pytorch_hp_example.rst>)
- [`PytorchModel` + MLflow](<project:../python_examples/ml/11_pytorch_mlflow_example.rst>)
- [`PytorchModel` freeze and fine-tune](<project:../python_examples/ml/12_pytorch_freeze_train.rst>)
- [`PytorchModel` with ResNet18](<project:../python_examples/ml/13_pytorch_resnet18.rst>)
