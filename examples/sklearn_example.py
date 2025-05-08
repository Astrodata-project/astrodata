from astrodata.models.SklearnModel import SklearnModel
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

model = SklearnModel()
model.config(model=LinearSVC, penalty="l2", loss="squared_hinge")

print(model.model)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
print(accuracy_score(y_test, preds))
