from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.metrics.regression import MAE, MSE, R2, RMSE
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.datasets import load_diabetes
import pandas as pd


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
print(model.get_metrics(X_test, y_test, metrics=[MAE(), MSE(), R2(), RMSE()]))