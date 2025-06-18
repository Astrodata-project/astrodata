import logging
import pprint

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector import (
    GridSearchCVSelector,
    GridSearchSelector,
)
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.models.XGBoostModel import XGBoostModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker

logging.basicConfig(level=logging.INFO)

############################
## Classification Example ##
############################


def classification_example():

    logging.info("Starting classification example...")

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    logging.info("Data split into training and test sets.")

    # Instantiate and configure the models
    skl_boost = SklearnModel(model_class=GradientBoostingClassifier)
    xgb_boost = XGBoostModel(
        model_class=XGBClassifier, tree_method="hist", enable_categorical=True
    )

    # Instantiate the metrics
    accuracy = SklearnMetric(accuracy_score)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss, greater_is_better=False)
    metrics = [accuracy, f1, logloss]

    # Define the search space for hyperparameters
    param_grid_skl = {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    }

    param_grid_xgb = {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "colsample_bytree": [0.8, 1.0],
    }

    # Prepare dictionary for both models
    models_params = {
        "skl": {
            "model": skl_boost,
            "param_grid": param_grid_skl,
        },
        "xgb": {
            "model": xgb_boost,
            "param_grid": param_grid_xgb,
        },
    }

    logging.info("Models and parameters defined.")

    results = {}

    for key, value in models_params.items():

        logging.info(f"Processing model: {key}")
        model = value["model"]
        param_grid = value["param_grid"]

        # Prepare the model_selectors
        gss = GridSearchSelector(
            model,
            val_size=0.2,
            param_grid=param_grid,
            scorer=logloss,
            metrics=metrics,
        )

        gss_cv = GridSearchCVSelector(
            model,
            cv=2,
            param_grid=param_grid,
            scorer=logloss,
            metrics=metrics,
        )

        # Fit both selectors
        gss.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=False)
        gss_cv.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=False)

        # Store results
        results[key] = {
            "par": gss.get_best_params(),
            "bm": gss.get_best_metrics(),
            "cv_par": gss_cv.get_best_params(),
            "cv_bm": gss_cv.get_best_metrics(),
        }

        logging.info(f"Model {key} processed successfully.")

    # Compare results
    results = pd.json_normalize(results).T
    logging.info("Results compiled successfully.")
    pprint.pprint(results)
    return results


########################
## Regression Example ##
#########################


def regression_example():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    # Instantiate and configure the models
    skl_boost = SklearnModel(model_class=GradientBoostingRegressor)
    xgb_boost = XGBoostModel(
        model_class=XGBRegressor, tree_method="hist", enable_categorical=True
    )
    # Instantiate the metrics
    mae = SklearnMetric(mean_absolute_error)
    mse = SklearnMetric(mean_squared_error)
    r2 = SklearnMetric(r2_score)
    metrics = [mae, mse, r2]
    # Define the search space for hyperparameters
    param_grid_skl = {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    }
    param_grid_xgb = {
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
        "colsample_bytree": [0.8, 1.0],
    }
    # Prepare dictionary for both models
    models_params = {
        "skl": {
            "model": skl_boost,
            "param_grid": param_grid_skl,
        },
        "xgb": {
            "model": xgb_boost,
            "param_grid": param_grid_xgb,
        },
    }
    results = {}
    for key, value in models_params.items():
        model = value["model"]
        param_grid = value["param_grid"]
        # Prepare the model_selectors
        gss = GridSearchSelector(
            model,
            val_size=0.2,
            param_grid=param_grid,
            scorer=mse,
            metrics=metrics,
        )
        gss_cv = GridSearchCVSelector(
            model,
            cv=2,
            param_grid=param_grid,
            scorer=mse,
            metrics=metrics,
        )
        # Fit both selectors
        gss.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=False)
        gss_cv.fit(X_train, y_train, X_test=X_test, y_test=y_test, verbose=False)
        # Store results
        results[key] = {
            "par": gss.get_best_params(),
            "bm": gss.get_best_metrics(),
            "cv_par": gss_cv.get_best_params(),
            "cv_bm": gss_cv.get_best_metrics(),
        }
        logging.info(f"Model {key} processed successfully.")
    # Compare results
    results = pd.json_normalize(results).T
    logging.info("Results compiled successfully.")
    pprint.pprint(results)
    return results


if __name__ == "__main__":
    classification_results = classification_example()
    print(classification_results)

    regression_results = regression_example()
    print(regression_results)
