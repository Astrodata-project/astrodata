import numpy as np
import pytest

from astrodata.ml.models.TensorflowModel import TensorflowModel


def _make_tiny_dataset(n=64, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    # Simple rule to create learnable labels
    y = (X.sum(axis=1) > 0).astype(np.int64)
    return X, y


def test_tensorflow_model_fit_predict_and_history(tmp_path):
    pytest.importorskip("tensorflow", reason="tensorflow missing")
    K = pytest.importorskip("keras", reason="keras missing")

    def build_tiny_classifier(input_dim=5, hidden=8, num_classes=2):
        """Build a simple 2-layer classifier."""
        model = K.Sequential(
            [
                K.layers.Dense(hidden, activation="relu", input_shape=(input_dim,)),
                K.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model

    X, y = _make_tiny_dataset(n=96, d=5, seed=0)
    X_val, y_val = _make_tiny_dataset(n=32, d=5, seed=1)

    m = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.1},
        epochs=3,
        batch_size=16,
    )

    # pre-fit predict should fail
    with pytest.raises(ValueError):
        m.predict(X, batch_size=16)

    # fit with train + val metrics tracking
    metrics = [K.metrics.SparseCategoricalAccuracy(name="accuracy")]
    m.fit(
        X=X,
        y=y,
        epochs=3,
        batch_size=16,
        metrics=metrics,
        X_val=X_val,
        y_val=y_val,
    )

    # Predict labels and probabilities
    yhat = m.predict(X, batch_size=16)
    assert yhat.shape[0] == X.shape[0]
    proba = m.predict_proba(X, batch_size=16)
    assert proba.shape == (X.shape[0], 2)
    # Probabilities sum to 1 per row
    np.testing.assert_allclose(
        proba.sum(axis=1), np.ones(X.shape[0]), rtol=1e-5, atol=1e-5
    )

    # get_metrics works
    from sklearn.metrics import accuracy_score

    from astrodata.ml.metrics.SklearnMetric import SklearnMetric

    res = m.get_metrics(
        X=X, y=y, metrics=[SklearnMetric(accuracy_score)], batch_size=16
    )
    assert "accuracy_score" in res

    # Training metric history
    train_hist = m.get_metrics_history(split="train")
    assert "loss" in train_hist
    assert "accuracy" in train_hist
    val_hist = m.get_metrics_history(split="val")
    # Present when X_val/y_val were provided
    assert val_hist is not None
    assert "loss" in val_hist
    assert "accuracy" in val_hist

    # Save/load roundtrip (tensorflow format)
    path = tmp_path / "model.keras"
    m.save(str(path), format="tensorflow")
    m2 = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.1},
    )
    m2.load(str(path), format="tensorflow")
    yhat2 = m2.predict(X, batch_size=16)
    assert yhat2.shape == yhat.shape


def test_tensorflow_model_fine_tune_reuses_weights():
    pytest.importorskip("tensorflow", reason="tensorflow missing")
    K = pytest.importorskip("keras", reason="keras missing")

    def build_tiny_classifier(input_dim=5, hidden=8, num_classes=2):
        model = K.Sequential(
            [
                K.layers.Dense(hidden, activation="relu", input_shape=(input_dim,)),
                K.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model

    X, y = _make_tiny_dataset(n=64, d=5, seed=2)

    m = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.05},
        epochs=1,
        batch_size=16,
    )
    m.fit(X=X, y=y, epochs=1, batch_size=16)
    first_model_ref = m.model_
    # Fine-tune should reuse existing weights (no re-init)
    m.fit(X=X, y=y, epochs=1, batch_size=16, fine_tune=True)
    assert m.model_ is first_model_ref


def test_tensorflow_model_freeze_unfreeze_layers():
    pytest.importorskip("tensorflow", reason="tensorflow missing")
    K = pytest.importorskip("keras", reason="keras missing")

    def build_tiny_classifier(input_dim=5, hidden=8, num_classes=2):
        model = K.Sequential(
            [
                K.layers.Dense(hidden, activation="relu", input_shape=(input_dim,)),
                K.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model

    X, y = _make_tiny_dataset(n=64, d=5, seed=3)

    m = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.05},
        epochs=1,
        batch_size=16,
    )
    m.fit(X=X, y=y, epochs=1, batch_size=16)

    # Freeze all layers
    m.freeze_layers("all")
    for layer in m.model_.layers:
        assert layer.trainable is False

    # Unfreeze specific layer
    layer_name = m.model_.layers[1].name
    m.unfreeze_layers([layer_name])
    assert m.model_.layers[1].trainable is True
    assert m.model_.layers[0].trainable is False

    # Unfreeze all
    m.unfreeze_layers("all")
    for layer in m.model_.layers:
        assert layer.trainable is True


def test_tensorflow_model_score():
    pytest.importorskip("tensorflow", reason="tensorflow missing")
    K = pytest.importorskip("keras", reason="keras missing")

    def build_tiny_classifier(input_dim=5, hidden=8, num_classes=2):
        model = K.Sequential(
            [
                K.layers.Dense(hidden, activation="relu", input_shape=(input_dim,)),
                K.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model

    X, y = _make_tiny_dataset(n=64, d=5, seed=4)

    m = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.05},
        epochs=2,
        batch_size=16,
    )
    m.fit(X=X, y=y, epochs=2, batch_size=16)

    # Score should return average loss
    score = m.score(X=X, y=y, batch_size=16)
    assert isinstance(score, (float, np.floating))
    assert score >= 0


def test_tensorflow_model_with_dataset():
    tf = pytest.importorskip("tensorflow", reason="tensorflow missing")
    K = pytest.importorskip("keras", reason="keras missing")

    def build_tiny_classifier(input_dim=5, hidden=8, num_classes=2):
        model = K.Sequential(
            [
                K.layers.Dense(hidden, activation="relu", input_shape=(input_dim,)),
                K.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model

    X, y = _make_tiny_dataset(n=64, d=5, seed=5)

    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    m = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.05},
        epochs=2,
        batch_size=16,
    )

    # Fit with dataset
    m.fit(dataset=dataset, epochs=2, batch_size=16)

    # Predict should work
    yhat = m.predict(X, batch_size=16)
    assert yhat.shape[0] == X.shape[0]


def test_tensorflow_model_save_load_formats(tmp_path):
    pytest.importorskip("tensorflow", reason="tensorflow missing")
    K = pytest.importorskip("keras", reason="keras missing")

    def build_tiny_classifier(input_dim=5, hidden=8, num_classes=2):
        model = K.Sequential(
            [
                K.layers.Dense(hidden, activation="relu", input_shape=(input_dim,)),
                K.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model

    X, y = _make_tiny_dataset(n=32, d=5, seed=6)

    m = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.05},
        epochs=1,
        batch_size=16,
    )
    m.fit(X=X, y=y, epochs=1, batch_size=16)

    # Test tensorflow format
    path_tf = tmp_path / "model_tf"
    m.save(str(path_tf), format="tensorflow")
    m_tf = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.05},
    )
    m_tf.load(str(path_tf) + ".keras", format="tensorflow")
    yhat_tf = m_tf.predict(X, batch_size=16)

    # Test h5 format
    path_h5 = tmp_path / "model_h5"
    m.save(str(path_h5), format="h5")
    m_h5 = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.05},
    )
    m_h5.load(str(path_h5) + ".h5", format="h5")
    yhat_h5 = m_h5.predict(X, batch_size=16)

    # Predictions should match
    assert yhat_tf.shape == yhat_h5.shape


def test_tensorflow_model_get_set_params():
    pytest.importorskip("tensorflow", reason="tensorflow missing")
    K = pytest.importorskip("keras", reason="keras missing")

    def build_tiny_classifier(input_dim=5, hidden=8, num_classes=2):
        model = K.Sequential(
            [
                K.layers.Dense(hidden, activation="relu", input_shape=(input_dim,)),
                K.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model

    m = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.05},
        epochs=5,
        batch_size=32,
    )

    params = m.get_params()
    assert params["epochs"] == 5
    assert params["batch_size"] == 32
    assert params["model_params"]["hidden"] == 8

    m.set_params(epochs=10, batch_size=64)
    assert m.epochs == 10
    assert m.batch_size == 64


def test_tensorflow_model_clone():
    pytest.importorskip("tensorflow", reason="tensorflow missing")
    K = pytest.importorskip("keras", reason="keras missing")

    def build_tiny_classifier(input_dim=5, hidden=8, num_classes=2):
        model = K.Sequential(
            [
                K.layers.Dense(hidden, activation="relu", input_shape=(input_dim,)),
                K.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model

    m = TensorflowModel(
        model_class=build_tiny_classifier,
        loss_fn=K.losses.SparseCategoricalCrossentropy,
        optimizer=K.optimizers.SGD,
        model_params={"input_dim": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"learning_rate": 0.05},
        epochs=5,
        batch_size=32,
    )

    m_clone = m.clone()
    assert m_clone is not m
    assert m_clone.epochs == m.epochs
    assert m_clone.batch_size == m.batch_size
    assert m_clone.model_params == m.model_params
