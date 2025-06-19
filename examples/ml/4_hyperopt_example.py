import pandas as pd
from hyperopt import hp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from astrodata.ml.model_selection.HyperOptSelector import HyperOptSelector
from astrodata.ml.models.SklearnModel import SklearnModel

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


space = {
    "C": hp.loguniform("C", -4, 2),  # valori tra 10^-4 e 10^2
    "dual": hp.choice("dual", [True, False]),
    "tol": hp.loguniform("tol", -6, -1),  # tipo 1e-6 a 1e-1
    "max_iter": hp.quniform("max_iter", 100, 2000, 100),
}


wrapped_model = SklearnModel(LinearSVC(penalty="l2", loss="squared_hinge"))

selector = HyperOptSelector(
    model=wrapped_model,  # attenzione poi c'è in HyperOptSelector  model.clone()
    cv=3,
    metrics=["accuracy"],  # tolgo?
)


selector = HyperOptSelector(
    model_class=LinearSVC, model_kwargs={"penalty": "l2", "loss": "squared_hinge"}
)

"""
gss = GridSearchCVSelector(
    model,
    param_grid={
        "C": [0.1, 1, 10],
    },
    scoring="accuracy",
    n_jobs=1,
    cv=3,
)

print(gss)

"""


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

"""
gss.fit(X_train, y_train)
print(gss.get_best_params())
print(gss.get_best_model())
print(type(gss.get_best_model()))
"""

selector.fit(X, y, space)

print(selector.get_params())
print(selector.fitted_model)
print(type(selector.get_best_model()))
