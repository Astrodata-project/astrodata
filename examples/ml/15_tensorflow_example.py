import os

import keras as K
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import TensorflowModel

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    class SimpleClassifier(K.Model):
        def __init__(self, **kwargs):
            super(SimpleClassifier, self).__init__()
            self.fc1 = K.layers.Dense(64, input_shape=(X_train.shape[1],))
            self.bn1 = K.layers.BatchNormalization()
            self.fc2 = K.layers.Dense(max(y_train))

        def call(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = K.activations.relu(x)
            x = self.fc2(x)
            return x

    model = TensorflowModel(
        model_class=SimpleClassifier,
        loss_fn=K.losses.BinaryCrossentropy,
        optimizer=K.optimizers.AdamW,
        optimizer_params={"learning_rate": 1e-3},
        epochs=10,
        batch_size=32,
    )
    print(model.get_params())

    accuracy = SklearnMetric(accuracy_score, greater_is_better=True)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss)

    metrics = [accuracy, f1, logloss]

    model.fit(
        X=X_train,
        y=y_train,
        device="gpu",
        seed=42,
    )

    y_pred = model.predict(
        data=X_test,
        batch_size=32,
    )

    print(model.get_metrics(X=X_test, y=y_test, metrics=metrics))
    print(model.score(X=X_test, y=y_test))
    print(model.get_metrics_history("val"))
    print(model.get_metrics_history("train"))
