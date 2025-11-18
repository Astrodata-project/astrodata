import glob
import json

import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import PytorchModel

if __name__ == "__main__":
    classes = json.load(open("testdata/imagenet_ex/imagenet_class_index.json"))
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    metrics = [SklearnMetric(accuracy_score, greater_is_better=True)]

    model = PytorchModel(
        model_class=resnet18(weights=weights),
        model_params={},
        loss_fn=nn.CrossEntropyLoss,
        optimizer=optim.AdamW,
        optimizer_params={"lr": 1e-3},
        epochs=10,
        batch_size=32,
        device="cpu",
        with_weight_init=True,
    )

    print(model)

    img_paths = sorted(glob.glob("testdata/imagenet_ex/*.jpg"))

    img_list = []

    for image_path in img_paths:
        img_list.append(transform(torchvision.io.read_image(image_path)))

    y_true = [242, 0]

    img_dataset = torch.utils.data.TensorDataset(
        torch.stack(img_list), torch.tensor(y_true, dtype=torch.long)
    )
    pred = model.predict(data=img_dataset, batch_size=1)

    dataset_val = torch.utils.data.TensorDataset(
        torch.stack(img_list), torch.tensor(y_true, dtype=torch.long)
    )

    pred = model.predict(data=img_dataset, batch_size=1)

    print(model.get_metrics(dataset=img_dataset, metrics=metrics))

    print(pred)

    for i in range(len(pred)):
        print(
            f"Ground Truth: {str.split(img_paths[i], '/')[-1]}    Prediction: {classes[str(pred[i])][1]}"
        )
