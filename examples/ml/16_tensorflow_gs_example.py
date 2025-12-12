import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
import keras as K
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.model_selection import GridSearchSelector
from astrodata.ml.models import TensorflowModel

if __name__ == "__main__":

    X, y = load_iris(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (X_train.astype("float32"), y_train.astype("int32"))
    )

    dataset_val = tf.data.Dataset.from_tensor_slices(
        (X_val.astype("float32"), y_val.astype("int32"))
    )

    def create_iris_model(input_dim, output_dim):
        """Create a simple Keras model for iris classification."""
        model = K.Sequential(
            [
                K.layers.Dense(16, activation="relu", input_shape=(input_dim,)),
                K.layers.Dense(output_dim, activation="softmax"),
            ]
        )
        return model

    model = TensorflowModel(
        model_class=create_iris_model,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.Adam,
        device=None,  # TensorFlow handles device automatically
    )

    print(model)

    param_grid = {
        "model_params": [{"input_dim": X.shape[1], "output_dim": 3}],
        "optimizer_params": [
            {"learning_rate": 1e-2},
            {"learning_rate": 1e-3},
            {"learning_rate": 1e-4},
        ],
        "batch_size": [32, 64],
        "epochs": [5, 10],
    }

    accuracy = SklearnMetric(accuracy_score, greater_is_better=True)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss)

    metrics = [accuracy, f1, logloss]

    gss = GridSearchSelector(
        model,
        param_grid=param_grid,
        scorer=accuracy,
        random_state=42,
        metrics=metrics,
    )

    gss.fit(dataset_train=dataset, dataset_val=dataset_val)

    print(gss.get_best_params())
    print(gss.get_best_metrics())
