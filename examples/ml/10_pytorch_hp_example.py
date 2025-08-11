from hyperopt import hp
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch import nn, optim

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import PytorchModel
from astrodata.ml.model_selection import HyperOptSelector

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    
    class IrisNet(nn.Module):
        def __init__(self, input_layers, output_layers):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_layers, 16),
                nn.ReLU(),
                nn.Linear(16, output_layers)
            )
        def forward(self, x):
            return self.layers(x)

    model = PytorchModel(
        model_class=IrisNet,
        loss_fn=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        device="cpu",
    )

    print(model)
    
    
    param_grid = {
        "model": hp.choice("model", [model]),
        "model_params":hp.choice("model_params", [{"input_layers": X.shape[1], "output_layers": 3}]),
        "optimizer_params": {"lr": hp.uniform("lr", 1e-4, 1e-2)},
        "batch_size":hp.choice("batch_size", [32, 64]),
        "epochs": hp.choice('epochs', [5,10,15])
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
    
    hos.fit(X, y)

    print(hos.get_best_params())
    print(hos.get_best_metrics())
