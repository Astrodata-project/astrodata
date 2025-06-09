import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models.SklearnModel import SklearnModel
from astrodata.ml.models.XGBoostModel import XGBoostModel

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
le = LabelEncoder()
y = le.fit_transform(y)

# Instantiate and configure the XGBoost model
xgb_model = XGBoostModel(
    model_class=XGBClassifier, tree_method="hist", enable_categorical=True
)

skl_model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")

models = [xgb_model, skl_model]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

for model in models:
    print(model)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(
        model.get_metrics(
            X_test,
            y_test,
            metrics=[
                SklearnMetric(accuracy_score),
                SklearnMetric(f1_score, average="micro"),
                SklearnMetric(confusion_matrix),
            ],
        )
    )
