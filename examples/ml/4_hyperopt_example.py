import pandas as pd
from hyperopt import hp
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.HyperOptSelector import (
    HyperOptSelector,
)  # ← Make sure the import path matches your project
from astrodata.ml.models.SklearnModel import SklearnModel

if __name__ == "__main__":
    # Load the breast cancer dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Instantiate the SklearnModel with LinearSVC and a metric
    model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")
    accuracy = SklearnMetric(accuracy_score)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss, greater_is_better=False)

    metrics = [accuracy, f1, logloss]

    # Define the hyperopt search space
    param_space = {
        "model": hp.choice("model", [model]),
        "C": hp.choice("C", [0.1, 1, 10]),
        "max_iter": hp.choice("max_iter", [1000, 2000]),
        "tol": hp.choice("tol", [1e-3, 1e-4]),
    }

    # Instantiate HyperOptSelector (using cross-validation in this example)
    hos = HyperOptSelector(
        param_space=param_space,
        scorer=accuracy,
        use_cv=True,
        cv=5,
        random_state=42,
        max_evals=100,  # You can increase this for a more thorough search
        metrics=metrics,
    )

    print(hos)

    hos.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    print(f"Best parameters found:", hos.get_best_params())
    print(f"Best metrics:", hos.get_best_metrics())
    print(f"Best model params:", hos.get_best_model().get_params())
