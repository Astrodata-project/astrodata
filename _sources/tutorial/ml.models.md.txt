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

```{attention}
To be implemented in future releases.
```

## `TensorflowModel`

```{attention}
To be implemented in future releases.
```

## Examples

- [Basic `SkLearnModel` usage](<project:../python_examples/ml/1_sklearn_example.rst>)
- [Multi `SkLearnModel` model training through `for` loops](<project:../python_examples/ml/2_multimodel_example.rst>)