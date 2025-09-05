import os
import tempfile

import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn, optim

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import PytorchModel

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    class SimpleClassifier(nn.Module):
        def __init__(self, input_layers, output_layers):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_layers, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, output_layers)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x

    model = PytorchModel(
        model_class=SimpleClassifier,
        model_params={
            "input_layers": X_train.shape[1],
            "output_layers": max(y_train) + 1,
        },
        loss_fn=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        optimizer_params={"lr": 1e-3},
        epochs=10,
        batch_size=32,
        device="cpu",
    )

    accuracy = SklearnMetric(accuracy_score, greater_is_better=True)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss)
    metrics = [accuracy, f1, logloss]

    model.fit(X=X_train, y=y_train)
    print("Model 1 metrics: ", model.get_metrics(X_test, y_test, metrics))

    # temporary safetensors file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    try:
        model.save(tmp_path, format="safetensors")

        model2 = PytorchModel(
            model_class=SimpleClassifier,
            model_params={
                "input_layers": X_train.shape[1],
                "output_layers": max(y_train) + 1,
            },
            loss_fn=nn.CrossEntropyLoss,
            optimizer=optim.AdamW,
            optimizer_params={"lr": 1e-3},
            epochs=10,
            batch_size=32,
            device="cpu",
        )

        model2.load(tmp_path, format="safetensors")
        print(
            "Is the loaded model equal to the original one?",
            model2.get_metrics(X_test, y_test, metrics)
            == model.get_metrics(X_test, y_test, metrics),
        )

        model2.freeze_layers(["fc1"])
        model2.fit(X=X_train, y=y_train, fine_tune=True)
        print("Model 2 metrics: ", model2.get_metrics(X_test, y_test, metrics))

    finally:
        os.remove(tmp_path)
