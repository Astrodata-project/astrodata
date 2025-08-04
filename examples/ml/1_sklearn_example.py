import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import SklearnModel

if __name__ == "__main__":

    # This example demonstrates how to use SklearnModel (or any astrodata.ml.models.BaseMlModel for that matter) for bsic tasks

    # Here we use the diabetes dataset, which is a regression dataset, skleadn.datasets.load_diabetes takes care of loading the data
    # and splitting it into features (X) and target (y). We then split the data into training and test sets using train_test_split from sklearn.model_selection.

    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # After loading the data, we instantiate a SklearnModel with the desired model class.
    # In this case, we use LinearSVR from sklearn.svm, which is a support vector regression model.
    # We also set a random state for reproducibility.
    # SklearnModel is a wrapper around sklearn models that provides compatibility with the astrodata.ml framework,
    # allowing you to use it seamlessly with the rest of the astrodata.ml ecosystem.
    # You can use any sklearn model class here, such as LinearRegression, RandomForestRegressor, etc.

    model = SklearnModel(model_class=LinearSVR, random_state=42)

    print(f"Model instantiated: {model}")

    # We can define the metrics we want to use for evaluation.
    # SklearnMetric is a wrapper around sklearn metrics that provides compatibility with the astrodata.ml framework.
    # Here we define several metrics commonly used for regression tasks, the greater_is_better parameter indicates whether a higher score is better for that metric.
    # For example, for mean_absolute_error, a lower value is better, so we set greater_is_better=False.

    mae = SklearnMetric(mean_absolute_error, greater_is_better=False)
    mse = SklearnMetric(mean_squared_error)
    r2 = SklearnMetric(r2_score, greater_is_better=True)
    msle = SklearnMetric(mean_squared_log_error)

    metrics = [mae, mse, r2, msle]

    # Now we can fit the model to the training data using the fit method.

    model.fit(X_train, y_train)

    # The predict method returns the predicted values for the test set.
    # preds = model.predict(X_test)
    # Here we show the metrics for the test set computed using the get_metrics method.

    metrics = model.get_metrics(
        X_test,
        y_test,
        metrics=metrics,
    )

    print(f"Metrics on test set: {metrics}")
