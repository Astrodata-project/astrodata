import torch
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn, optim

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import PytorchModel
from astrodata.tracking.MLFlowTracker import PytorchMLflowTracker

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    dataset_val = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)
    )

    dataset_test = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

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

    print(model.get_params())

    tracker = PytorchMLflowTracker(
        run_name="MlFlowWithVal",
        experiment_name="11_pytorch_mlflow_example.py",
        extra_tags={"stage": "testing"},
    )

    tracked_model = tracker.wrap_fit(
        model,
        dataset_val=dataset_val,
        dataset_test=dataset_test,
        metrics=metrics,
        log_model=True,
    )

    tracked_model.fit(dataset_train=dataset)

    y_pred = tracked_model.predict(
        X=X_test,
        batch_size=32,
    )

    print(
        "Test metrics:",
        tracked_model.get_metrics(dataset=dataset_test, metrics=metrics),
    )
