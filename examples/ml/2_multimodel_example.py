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

# This example shows how to use different models from the astrodata.ml.models package in the same for loop.

if __name__ == "__main__":

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Here both models are instantiated with their respective model classes.

    xgb_model = XGBoostModel(
        model_class=XGBClassifier, tree_method="hist", enable_categorical=True
    )

    skl_model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")

    models = [skl_model, xgb_model]

    # Here we define the metrics we want to use for evaluation; we can see in f1 that a kwarg "avberage" is passed, which is specific to the f1_score function.

    accuracy = SklearnMetric(accuracy_score)
    f1 = SklearnMetric(f1_score, average="micro")

    metrics = [accuracy, f1]

    # Once everything is set up, we can loop through the models, fit them to the training data, and evaluate their performance on the test set.

    for model in models:
        print(f"Model instantiated: %s", model)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        res = model.get_metrics(X_test, y_test, metrics=metrics)

        print(f"Metrics for model %s: %s", model, res)
