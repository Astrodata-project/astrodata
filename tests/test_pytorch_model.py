import numpy as np
import pandas as pd
import pytest

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.models.PytorchModel import PytorchModel


def _make_tiny_dataset(n=64, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    # Simple rule to create learnable labels
    y = (X.sum(axis=1) > 0).astype(np.int64)
    return X, y


def test_pytorch_model_fit_predict_and_history(tmp_path):
    torch = pytest.importorskip("torch", reason="torch missing")
    nn = torch.nn
    optim = torch.optim
    from sklearn.metrics import accuracy_score

    # Tiny 2-layer classifier with 2 outputs
    class TinyNet(nn.Module):
        def __init__(self, in_features=5, hidden=8, num_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    X, y = _make_tiny_dataset(n=96, d=5, seed=0)
    X_val, y_val = _make_tiny_dataset(n=32, d=5, seed=1)

    m = PytorchModel(
        model_class=TinyNet,
        loss_fn=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        model_params={"in_features": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"lr": 0.1},
        device="cpu",
        epochs=3,
        batch_size=16,
    )

    # pre-fit predict should fail
    with pytest.raises(RuntimeError):
        m.predict(X, batch_size=16)

    # fit with train + val metrics tracking
    metrics = [SklearnMetric(accuracy_score)]
    m.fit(
        X=X,
        y=y,
        epochs=3,
        batch_size=16,
        device="cpu",
        metrics=metrics,
        X_val=X_val,
        y_val=y_val,
    )

    # Predict labels and probabilities
    yhat = m.predict(X, batch_size=16, device="cpu")
    assert yhat.shape[0] == X.shape[0]
    proba = m.predict_proba(X, batch_size=16, device="cpu")
    assert proba.shape == (X.shape[0], 2)
    # Probabilities sum to 1 per row
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(X.shape[0]), rtol=1e-5, atol=1e-5)

    # get_metrics works
    res = m.get_metrics(X, y, metrics=metrics, batch_size=16, device="cpu")
    assert "accuracy_score" in res

    # Training metric history (per step) and validation metric history (per epoch)
    train_hist = m.get_metrics_history(split="train")
    assert "accuracy_score_epoch" in train_hist
    assert "loss_epoch" in train_hist
    val_hist = m.get_metrics_history(split="val")
    # Present when X_val/y_val were provided
    assert val_hist is not None
    # Key suffixed with _val_epoch exists for validation metric
    assert any(k.endswith("_epoch") for k in val_hist.keys())

    # Save/load roundtrip (torch format)
    path = tmp_path / "model.pt"
    m.save(str(path), format="torch")
    m2 = PytorchModel(
        model_class=TinyNet,
        loss_fn=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        model_params={"in_features": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"lr": 0.1},
        device="cpu",
    )
    m2.load(str(path), format="torch")
    yhat2 = m2.predict(X, batch_size=16, device="cpu")
    assert yhat2.shape == yhat.shape


def test_pytorch_model_fine_tune_reuses_weights():
    torch = pytest.importorskip("torch", reason="torch missing")
    nn = torch.nn
    optim = torch.optim
    from sklearn.metrics import accuracy_score

    class TinyNet(nn.Module):
        def __init__(self, in_features=5, hidden=8, num_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    X, y = _make_tiny_dataset(n=64, d=5, seed=2)

    m = PytorchModel(
        model_class=TinyNet,
        loss_fn=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        model_params={"in_features": 5, "hidden": 8, "num_classes": 2},
        optimizer_params={"lr": 0.05},
        device="cpu",
        epochs=1,
        batch_size=16,
    )
    m.fit(X=X, y=y, epochs=1, batch_size=16, device="cpu", metrics=[SklearnMetric(accuracy_score)])
    first_model_ref = m.model_
    # Fine-tune should reuse existing weights (no re-init)
    m.fit(X=X, y=y, epochs=1, batch_size=16, device="cpu", metrics=[SklearnMetric(accuracy_score)], fine_tune=True)
    assert m.model_ is first_model_ref

    # Freezing layers unfreezes only selected
    m.freeze_layers(["net.2"])  # unfreeze last linear only
    # All others should be frozen
    for name, param in m.model_.named_parameters():
        if name.startswith("net.2"):
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

