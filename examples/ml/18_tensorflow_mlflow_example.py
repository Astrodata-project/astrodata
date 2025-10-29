import keras as K
import tensorflow as tf
from keras.metrics import SparseCategoricalAccuracy
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from astrodata.ml.metrics import TensorflowMetric
from astrodata.ml.models import TensorflowModel
from astrodata.tracking.MLFlowTracker import TensorflowMLflowTracker

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Convert to float32 for TensorFlow and ensure proper data types
    X_train = X_train.astype("float32")
    X_val = X_val.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("int32")
    y_val = y_val.astype("int32")
    y_test = y_test.astype("int32")

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    dataset_val = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def create_classifier(input_dim, output_dim):
        """Create a simple Keras model for binary classification."""
        model = K.Sequential(
            [
                K.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
                K.layers.BatchNormalization(),
                K.layers.Dense(
                    output_dim, activation="sigmoid" if output_dim == 1 else "softmax"
                ),
            ]
        )
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
        device=None,  # TensorFlow handles device automatically
    )

    accuracy = TensorflowMetric(SparseCategoricalAccuracy())

    metrics = [accuracy]

    print(model.get_params())

    tracker = TensorflowMLflowTracker(
        run_name="MlFlowWithVal",
        experiment_name="18_tensorflow_mlflow_example.py",
        extra_tags={"stage": "testing"},
    )

    tracked_model = tracker.wrap_fit(
        model,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        metrics=metrics,
        log_model=True,
    )

    tracked_model.fit(dataset=dataset)

    y_pred = tracked_model.predict(
        data=X_test,
        batch_size=32,
    )

    print(
        "Test metrics:",
        tracked_model.get_metrics(dataset=dataset_test, metrics=metrics),
    )
