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

    torch_model = SimpleClassifier(X_train.shape[1], max(y_train) + 1)
    optimizer = optim.AdamW(torch_model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    model = PytorchModel(
        torch_model=torch_model,
        loss_fn=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    print(model)

    model.fit(
        X=X_train,
        y=y_train,
        epochs=10,
        batch_size=32,
    )

    y_pred = model.predict(
        X=X_test,
        batch_size=32,
    )

    accuracy = SklearnMetric(accuracy_score, greater_is_better=True)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss)

    metrics = [accuracy, f1, logloss]

    print(model.get_metrics(X_test, y_test, metrics))
