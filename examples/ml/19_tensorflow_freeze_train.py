import os
import tempfile

import keras as K
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import TensorflowModel

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Convert to proper types for TensorFlow
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("int32")
    y_test = y_test.astype("int32")

    def create_classifier(input_dim, output_dim):
        """Create a simple Keras model for binary classification."""
        inputs = K.layers.Input(shape=(input_dim,))
        x = K.layers.Dense(64, activation="relu", name="fc1")(inputs)
        x = K.layers.BatchNormalization(name="bn1")(x)
        outputs = K.layers.Dense(
            output_dim,
            activation="sigmoid" if output_dim == 1 else "softmax",
            name="fc2",
        )(x)

        model = K.Model(inputs=inputs, outputs=outputs, name="SimpleClassifier")
        return model

    model = TensorflowModel(
        model_class=create_classifier,
        model_params={
            "input_dim": X_train.shape[1],
            "output_dim": max(y_train) + 1,
        },
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.Adam,
        optimizer_params={"learning_rate": 1e-3},
        epochs=10,
        batch_size=32,
        device=None,
    )

    accuracy = SklearnMetric(accuracy_score, greater_is_better=True)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss)
    metrics = [accuracy, f1, logloss]

    model.fit(X=X_train, y=y_train)
    print("Model 1 metrics: ", model.get_metrics(X=X_test, y=y_test, metrics=metrics))

    # temporary tensorflow file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        model.save(tmp_path, format="tensorflow")

        model2 = TensorflowModel(
            model_class=create_classifier,
            model_params={
                "input_dim": X_train.shape[1],
                "output_dim": max(y_train) + 1,
            },
            loss_fn=K.losses.SparseCategoricalCrossentropy,
            optimizer=K.optimizers.Adam,
            optimizer_params={"learning_rate": 1e-3},
            epochs=10,
            batch_size=32,
            device=None,
        )

        model2.load(tmp_path, format="tensorflow")
        print(
            "Is the loaded model equal to the original one?",
            model2.get_metrics(X=X_test, y=y_test, metrics=metrics)
            == model.get_metrics(X=X_test, y=y_test, metrics=metrics),
        )

        model2.freeze_layers("all")
        model2.unfreeze_layers(["fc2"])
        model2.fit(X=X_train, y=y_train, fine_tune=True)
        print(
            "Model 2 metrics: ", model2.get_metrics(X=X_test, y=y_test, metrics=metrics)
        )

    finally:
        os.remove(tmp_path)
