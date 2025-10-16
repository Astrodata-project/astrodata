import keras as K
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import TensorflowModel

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    class SimpleClassifier(K.Model):
        def __init__(self, input_layers, output_layers):
            super(SimpleClassifier, self).__init__()
            self.fc1 = K.layers.Dense(64, input_shape=(input_layers,))
            self.bn1 = K.layers.BatchNormalization()
            self.fc2 = K.layers.Dense(output_layers)

        def call(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = K.activations.relu(x)
            x = self.fc2(x)
            return x

    model = TensorflowModel(
        model_class=SimpleClassifier,
        model_params={
            "input_layers": X_train.shape[1],
            "output_layers": max(y_train),
        },
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
    )

    y_pred = model.predict(
        X=X_test,
        batch_size=32,
    )

    print(model.get_metrics(X_test, y_test, metrics))
    print(model.get_metrics_history("val"))
    print(model.get_metrics_history("train"))
