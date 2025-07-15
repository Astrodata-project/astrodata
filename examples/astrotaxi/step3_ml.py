"""
Astrodata HyperOptSelector Tutorial with MLflow Tracking
=======================================================

This example demonstrates how to perform hyperparameter optimization using
the `HyperOptSelector` from astrodata, comparing two popular regression models:
RandomForestRegressor and XGBRegressor. Experiment tracking is handled by MLflow.

We will:
    - Define two different search spaces (one for each regressor)
    - Use custom metrics for evaluation
    - Track the experiments and best models with MLflow
"""

from hyperopt import hp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from xgboost import XGBRegressor

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.HyperOptSelector import HyperOptSelector
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.models.XGBoostModel import XGBoostModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker


def run_hyperopt_example(X_train, y_train, X_test, y_test):
    # 1. Define the models to be optimized
    # Wrap the sklearn and xgboost models with their respective astrodata wrappers.
    randomforest = SklearnModel(model_class=RandomForestRegressor)
    xgboost = XGBoostModel(model_class=XGBRegressor)

    # 2. Set up the experiment tracker using MLflow
    # All training runs, metrics, and best models will be logged under this MLflow experiment.
    tracker = SklearnMLflowTracker(
        run_name="AstroTaxi",  # Name of this run in MLflow
        experiment_name="examples_astrotaxi.py",  # Name of the MLflow experiment
        extra_tags=None,  # Extra tags for MLflow (optional)
    )

    # 3. Define evaluation metrics
    # Each metric is wrapped using SklearnMetric for astrodata compatibility.
    # Set greater_is_better=False for metrics where a lower value is best.
    MAE = SklearnMetric(metric=mean_absolute_error, greater_is_better=False)
    MSE = SklearnMetric(metric=mean_squared_error, greater_is_better=False)
    MSLE = SklearnMetric(metric=mean_squared_log_error, greater_is_better=False)
    R2 = SklearnMetric(metric=r2_score, greater_is_better=True)
    metrics = [MAE, MSE, MSLE, R2]

    # 4. Define the hyperparameter search spaces for each model
    # For RandomForest: search over n_estimators and max_depth
    param_space_randomforest = {
        "model": hp.choice(
            "model", [randomforest]
        ),  # Model is included in the search space
        "n_estimators": hp.choice("n_estimators", [50, 100]),
        "max_depth": hp.choice("max_depth", [3, 5, 7]),
    }

    # For XGBoost: search over n_estimators and learning_rate
    param_space_xgboost = {
        "model": hp.choice("model", [xgboost]),
        "n_estimators": hp.choice("n_estimators", [50, 100]),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
    }

    # 5. Run hyperparameter optimization for both models
    # Loop over both parameter spaces, running a separate HyperOptSelector for each.
    for param_space in [param_space_randomforest, param_space_xgboost]:
        # Create the HyperOptSelector.
        # - param_space: search space for hyperopt
        # - scorer: metric to optimize (here, R2)
        # - use_cv: whether to use cross-validation (False = simple split)
        # - random_state: seed for reproducibility
        # - max_evals: number of hyperopt trials
        # - metrics: additional metrics to record
        # - tracker: MLflow tracker

        print(f"Running HyperOpt for model with param space: {param_space}")

        hos = HyperOptSelector(
            param_space=param_space,
            scorer=R2,
            use_cv=False,  # Set to True to use cross-validation
            random_state=42,
            max_evals=10,  # Number of hyperopt search iterations
            metrics=metrics,
            tracker=tracker,
        )

        # Fit the selector. This runs the hyperparameter search.
        hos.fit(X_train, y_train, X_test=X_test, y_test=y_test)

        # Print best parameters and metrics found during the search
        print("Best parameters found: ", hos.get_best_params())
        print("Best metrics: ", hos.get_best_metrics())

    # 6. Register the best model in MLflow for production
    # - metric: which metric to use for selecting the best run (should match scorer)
    # - split_name: which dataset split to evaluate ("val" or "test")
    # - stage: production stage name in MLflow Model Registry
    tracker.register_best_model(
        metric=R2,
        split_name="val",
        stage="Production",
    )
