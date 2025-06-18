import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector import (
    GridSearchCVSelector,
    GridSearchSelector,
)
from astrodata.ml.models.SklearnModel import SklearnModel

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")
# model = XGBoostModel(model_class=XGBClassifier, tree_method="hist", enable_categorical=True)

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

# gss = GridSearchSelector(
#    model,
#    param_grid={
#        "C": [0.1, 1, 10],
#        "max_iter": [1000, 2000],
#        "tol": [1e-3, 1e-4],
#    },
#    scorer=accuracy,
#    val_size=0.2,
#    random_state=42,
#    metrics=None,
# )

print(gss)


gss.fit(X_train, y_train, X_test=X_test, y_test=y_test)
print(gss.get_best_params())
print(gss.get_best_model())
print(type(gss.get_best_model()))
