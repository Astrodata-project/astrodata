# model selection

The `astrodata.ml.model_selection` module provides tools for systematically finding the best machine learning model and hyperparameters for a given task using different heuristics or strategies.

## Abstract Class

**`BaseMlModelSelector`** is the abstract base class of any model selection strategy. Subclasses must implement:
* `fit(X, y, *args, **kwargs)`: Runs the model selection process on the data.
* `get_best_model()`: Returns the best `BaseMlModel` instance found.
* `get_best_params()`: Returns the hyperparameters of the best model.
* `get_best_metrics()`: Returns the evaluation metrics for the best model.
* `get_params()`: Returns the parameters of the selector itself.

## How to Use

### Initializing 

Initialization depends on the selector that is being used; generally, a model selector is initialized with a model to perform the search on and a grid of parameters to test. 

```python
from astrodata.ml.model_selection.GridSearchSelector import GridSearchCVSelector

gss = GridSearchCVSelector(
        model=model,
        #tracker=tracker,
        param_grid={
            "C": [0.1, 1, 10],
            "max_iter": [1000, 2000],
            "tol": [1e-3, 1e-4],
        },
        scorer=accuracy,
        cv=5,
        random_state=42,
        metrics=None,
    )
```

The `scorer` parameter of the selector is a `BaseMetric` and it is used to decide what model is the best by computing said metric and using it as a discriminator. Optionally, a list of metrics can be passed as an argument to compute said metrics at each step and at the end (this is particularly relevant when a `tracker` is added as those metrics will be saved in MlFlow, check [`this section`](<project:./tracking.models.md>) for more info).


```{attention}
Depending on the chosen model selector, the `param_grid` may change.
```

After a selector is initialized, the next step is to `fit` it to a set of data, doing so the selector tries all the required combinations and finally fits the model whose parameters returned the best results.

```python
best_model =  gss.fit(X_train, y_train)

print(f"Best parameters found: {gss.get_best_params()}")
print(f"Best metrics: {gss.get_best_metrics()}")
print(f"Best model: {best_model.get_params()}")
```

## `GridSearchSelector`

Implements an exhaustive search over a specified parameter grid. It trains and evaluates models for every combination of hyperparameters, selecting the one that performs best according to a given `scorer`. It supports both single validation split and cross-validation if using `GridSearchCVSelector`.

### Parameters
* **model** : BaseMlModel
  * The model to optimize.
* **param_grid** : dict
  * Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
* **scorer** : BaseMetric, optional
  * The metric used to select the best model. If None, model's default score method is used.
* **val_size** (for GridSearchSelector): float, optional (default None)
  * Fraction of training data to use as validation split.
* **cv** (for GridSearchCVSelector): int or cross-validation splitter (default=5)
  * Number of folds (int) or an object that yields train/test splits.
* **random_state** : int, optional
  * Random seed for reproducibility.
* **metrics** : list of BaseMetric, optional
  * Additional metrics to evaluate on validation set.
* **tracker** : ModelTracker, optional
  * Optional experiment/model tracker for logging.
* **log_all_models** : bool, optional
  * If True, logs all models to the tracker, not just the best one.


## `HyperOptSelector`

Utilizes the [`hyperopt`](https://hyperopt.github.io/hyperopt/) library for efficient hyperparameter optimization. Instead of exhaustive search, `hyperopt` uses Bayesian optimization (Tree-structured Parzen Estimator, TPE) to intelligently explore the parameter space, often finding better results with fewer evaluations compared to traditional grid search. It requires a `param_space` defined using `hyperopt.hp` functions.

```python
# Define the hyperopt search space
param_space = {
    "model": hp.choice("model", [model]),
    "C": hp.choice("C", [0.1, 1, 10]),
    "max_iter": hp.choice("max_iter", [1000, 2000]),
    "tol": hp.choice("tol", [1e-3, 1e-4]),
}
```

### Parameters

* **param_grid** : dict
    * Dictionary with parameter search spaces as shown [here](https://hyperopt.github.io/hyperopt/-started/search_spaces/).
* **scorer** : BaseMetric, optional
    * The metric used to select the best model. If None, model's default score method is used.
* **use_cv**: bool
    * Wether to use cross validation or regular validation split.
* **cv** : int or cross-validation splitter (default=5)
    * Number of folds (int) or an object that yields train/test splits.
* **max_evals**: int
    * Maximum number of evaluations hyperopt can run.
* **random_state** : int, optional
    * Random seed for reproducibility.
* **metrics** : list of BaseMetric, optional
    * Additional metrics to evaluate on validation folds.
* **tracker** : ModelTracker, optional
    * Optional experiment/model tracker for logging.
* **log_all_models** : bool, optional
    * If True, logs all models, not just the best one.

## Examples

- [Basic `GridSearchSelector` usage](<project:../python_examples/ml/3_gridsearch_example.rst>)
- [Basic `HyperOptSelector` usage](<project:../python_examples/ml/4_hyperopt_example.rst>)
- [`GridSearchSelector` with MlFlow tracking](<project:../python_examples/ml/6_mlflow_gs_example.rst>)
- [`HyperOptSelector` with MlFlow tracking](<project:../python_examples/ml/7_mlflow_hp_example.rst>)
