from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.model_selection.GridSearchSelector import GridSearchSelector
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the Iris dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")

gss = GridSearchSelector(
    model=model,
    param_grid={
        'C': [0.1, 1, 10],
        'max_iter': [100, 200]
    },
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1,
    refit=True
)

print(gss)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

gss.fit(X_train, y_train)
print(gss.get_best_params())
print(gss.get_best_model())
