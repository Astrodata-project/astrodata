from hyperopt import hp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from xgboost import XGBRegressor

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.HyperOptSelector import HyperOptSelector
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.models.XGBoostModel import XGBoostModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker


def run_hyperopt_example(X_train, y_train, X_test, y_test):

    randomforest = SklearnModel(model_class=RandomForestRegressor)
    xgboost = XGBoostModel(model_class=XGBRegressor)

    tracker = SklearnMLflowTracker(
        run_name="AstroTaxi",
        experiment_name="examples_astrotaxi.py",
        extra_tags=None,
    )

    # Define the metrics to be used for evaluation

    MAE = SklearnMetric(metric=mean_absolute_error, greater_is_better=False)
    MSE = SklearnMetric(metric=mean_squared_error, greater_is_better=False)
    MSLE = SklearnMetric(metric=mean_squared_log_error, greater_is_better=False)
    R2 = SklearnMetric(metric=r2_score, greater_is_better=True)

    metrics = [MAE, MSE, MSLE, R2]

    param_space_randomforest = {
        "model": hp.choice("model", [randomforest]),
        "n_estimators": hp.choice("n_estimators", [50, 100, 200]),
        "max_depth": hp.choice("max_depth", [3, 5, 7]),
    }

    param_space_xgboost = {
        "model": hp.choice("model", [xgboost]),
        "n_estimators": hp.choice("n_estimators", [50, 100, 200]),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3)
    }

    for param_space in [param_space_randomforest, param_space_xgboost]:

        # Instantiate HyperOptSelector (using cross-validation in this example)
        hos = HyperOptSelector(
            param_space=param_space,
            scorer=R2,
            use_cv=False,
            random_state=42,
            max_evals=10,
            metrics=metrics,
            tracker=tracker,
        )

        hos.fit(X_train, y_train, X_test=X_test, y_test=y_test)

        print("Best parameters found: ", hos.get_best_params())
        print("Best metrics: ", hos.get_best_metrics())

    # Here we tag for production the best model found during the grid search. The experiments in mlflow
    # are organized by the specified metric and the best performing one is registered.
    # make sure to use the same metric as the one used as scorer in the GridSearchCVSelector.

    tracker.register_best_model(
        metric=R2,
        split_name="val",
        stage="Production",
    )
