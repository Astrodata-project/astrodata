import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn, optim
from torchvision import datasets

from astrodata.ml.models import PytorchModel

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Create datasets for training & validation, download if necessary
    training_set = datasets.FashionMNIST(
        "./testdata", train=True, transform=transform, download=True
    )
    validation_set = datasets.FashionMNIST(
        "./testdata", train=False, transform=transform, download=True
    )

    print("Training set has {} instances".format(len(training_set)))

    class GarmentClassifier(nn.Module):
        def __init__(self):
            super(GarmentClassifier, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    torch_model = GarmentClassifier()
    optimizer = optim.SGD(torch_model.parameters(), lr=0.01, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()

    model = PytorchModel(
        torch_model=torch_model,
        loss_fn=loss_function,
        optimizer=optimizer,
    )

    print(model)
    X = training_set.data.float()
    y = training_set.targets.long()

    X_val = validation_set.data.float().unsqueeze(1)
    X = X.unsqueeze(1)

    model.fit(
        X=X,
        y=y,
        epochs=5,
        batch_size=64,
    )

    y_pred = model.predict(
        X=X_val,
        validation_set=validation_set.data.float().to("cuda"),
        batch_size=64,
    )

    y_val = validation_set.targets.long()

    print(sum(y_val == y_pred) / len(y_pred))
