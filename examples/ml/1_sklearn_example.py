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

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models.SklearnModel import SklearnModel

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

model = SklearnModel(model_class=LinearSVR, random_state=42)

print(model)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model.fit(X_train, y_train)


preds = model.predict(X_test)
print(
    model.get_metrics(
        X_test,
        y_test,
        metrics=[
            SklearnMetric(mean_absolute_error),
            SklearnMetric(mean_squared_error),
            SklearnMetric(r2_score),
            SklearnMetric(mean_squared_log_error),
        ],
    )
)
