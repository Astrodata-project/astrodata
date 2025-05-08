import pandas as pd
from astrodata.models.SklearnModel import SklearnModel
from astrodata.models.XgBoostModel import XGBoostModel
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from xgboost import XGBClassifier

# Load the Iris dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
le = LabelEncoder()
y = le.fit_transform(y)

# Instantiate and configure the XGBoost model
xgb_model = XGBoostModel()
xgb_model.config(model=XGBClassifier, tree_method="hist", enable_categorical=True)

skl_model = SklearnModel()
skl_model.config(model=LinearSVC, penalty="l2", loss="squared_hinge")

models = [xgb_model, skl_model]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

for model in models:
    print(model.model)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(accuracy_score(y_test, preds))