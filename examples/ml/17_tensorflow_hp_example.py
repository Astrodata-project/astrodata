import keras as K
import tensorflow as tf
from hyperopt import hp
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.model_selection import HyperOptSelector
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
        "model": hp.choice("model", [model]),
        "model_params": hp.choice(
            "model_params", [{"input_dim": X.shape[1], "output_dim": 3}]
        ),
        "optimizer_params": {"learning_rate": hp.uniform("learning_rate", 1e-4, 1e-2)},
        "batch_size": hp.choice("batch_size", [32, 64]),
        "epochs": hp.choice("epochs", [5, 10, 15]),
    }

    accuracy = SklearnMetric(accuracy_score, greater_is_better=True)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss)

    metrics = [accuracy, f1, logloss]

    hos = HyperOptSelector(
        param_space=param_grid,
        scorer=accuracy,
        use_cv=False,
        val_size=0.1,
        random_state=42,
        max_evals=100,
        metrics=metrics,
    )

    hos.fit(dataset_train=dataset, dataset_val=dataset_val)

    print(hos.get_best_params())
    print(hos.get_best_metrics())
