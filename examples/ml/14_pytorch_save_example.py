import os
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn, optim

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import PytorchModel


if __name__ == "__main__":
    # Load data
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define a simple neural network
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, output_dim)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            return self.fc2(x)

    # Instantiate model wrapper
    model = PytorchModel(
        model_class=SimpleClassifier,
        model_params={"input_dim": X_train.shape[1], "output_dim": len(set(y_train))},
        loss_fn=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        optimizer_params={"lr": 1e-3},
        epochs=10,
        batch_size=32,
        device="cpu",
    )

    # Ensure checkpoint folder exists
    checkpoint_dir = "testdata/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Fit model and save checkpoint every 2 epochs
    model.fit(
        X=X_train,
        y=y_train,
        save_every_n_epochs=2,
        save_folder=checkpoint_dir,
    )

    # Evaluate on test set
    y_pred = model.predict(X=X_test, batch_size=32)

    # Define metrics
    accuracy = SklearnMetric(accuracy_score, greater_is_better=True)
    f1 = SklearnMetric(f1_score, average="micro")
    logloss = SklearnMetric(log_loss)
    metrics = [accuracy, f1, logloss]

    # Print results
    print(model.get_metrics(X_test, y_test, metrics))

    # Load last checkpoint and evaluate
    ckpt_path = os.path.join(checkpoint_dir, "checkpoint_8.pt")
    print("Loading checkpoint from", ckpt_path)
    model.load(ckpt_path)
    print("Checkpoint model metrics:", model.get_metrics(X_test, y_test, metrics))
