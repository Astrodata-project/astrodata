import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.model_selection import GridSearchCVSelector
from astrodata.ml.models import SklearnModel

if __name__ == "__main__":
    # This example shows how to use the GridSearchCVSelector with a SklearnModel.
    # It performs hyperparameter tuning on a LinearSVC model using cross-validation.

    # Load the breast cancer dataset

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Instantiate the SklearnModel with LinearSVC and a metric to use as a scorer for the grid search.
    # the scorer determines how the model's performance is evaluated during hyperparameter tuning.

    model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")

    accuracy = SklearnMetric(accuracy_score, greater_is_better=True)

    gss = GridSearchCVSelector(
        model,
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

    print(gss)

    gss.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    print(f"Best parameters found: {gss.get_best_params()}")
    print(f"Best metrics: {gss.get_best_metrics()}")
    print(f"Best model: {gss.get_best_model().get_params()}")
