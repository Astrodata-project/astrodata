import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics import accuracy_score

from astrodata.data.loaders.torch_loader import TorchLoader
from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.models import PytorchModel
from astrodata.tracking.MLFlowTracker import PytorchMLflowTracker
from testdata import download_galaxy_mnist

# ============================================================
# DATA LOADING
# ============================================================
data_path = download_galaxy_mnist()
print(f"GalaxyMNIST data downloaded and extracted to: {data_path}")

img_size = (64, 64)

# Define data augmentation transforms
train_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),  # If starting from numpy/tensor
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts to float32 and scales [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

generic_transforms = transforms.Compose(
    [
        transforms.ToPILImage(),  # If starting from numpy/tensor
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # Converts to float32 and scales [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_dict = {
    "train": train_transforms,
    "test": generic_transforms,
    "val": generic_transforms,
}

loader = TorchLoader(transform_dict=transform_dict)

galaxymnist_data = loader.load(data_path)

galaxymnist_train = galaxymnist_data.get_dataset("train")
galaxymnist_test = galaxymnist_data.get_dataset("test")
galaxymnist_val = galaxymnist_data.get_dataset("val")

# Display class mappings for both splits
print(f"GalaxyMNIST train class_to_idx: {galaxymnist_train.class_to_idx}")
print(f"GalaxyMNIST test  class_to_idx: {galaxymnist_test.class_to_idx}")

num_classes = len(galaxymnist_train.class_to_idx)
print(f"Number of classes: {num_classes}")

# ============================================================
# MODEL ARCHITECTURE
# ============================================================


# Define custom ResNet50 model with classification head
class GalaxyMNISTResNet50(nn.Module):
    """ResNet50 model for GalaxyMNIST classification with custom head."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super(GalaxyMNISTResNet50, self).__init__()

        # Load pre-trained ResNet50 and remove final classification layer
        base_model = models.resnet50(pretrained=pretrained)
        self.resnet50 = nn.Sequential(
            *list(base_model.children())[:-2]
        )  # Remove avgpool and fc

        # Custom classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),  # ResNet50 outputs 2048 features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """Forward pass through the model."""
        # ResNet50 feature extraction
        features = self.resnet50(x)
        # Global average pooling
        pooled = self.avgpool(features)
        # Classification head
        output = self.classifier(pooled)
        return output


# Instantiate the model
galaxy_mnist_resnet = GalaxyMNISTResNet50(num_classes=num_classes, pretrained=True)

print(f"Model created with {num_classes} output classes")
print("Model summary:")
print(galaxy_mnist_resnet)

# ============================================================
# METRICS AND TRACKING SETUP
# ============================================================
accuracy_metric = SklearnMetric(accuracy_score, name="accuracy", greater_is_better=True)
metrics = [accuracy_metric]

tracker = PytorchMLflowTracker(
    run_name="GalaxyMNIST_ResNet50_FineTuning",
    experiment_name="GalaxyMNIST_ResNet50_FineTuning",
    extra_tags={"stage": "testing"},
)
# ============================================================
# PHASE 1: TRAIN CUSTOM HEAD ONLY
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Training custom classification head")
print("Base ResNet50 layers are frozen")
print("=" * 60 + "\n")

model_phase1 = PytorchModel(
    model_class=galaxy_mnist_resnet,
    model_params={},
    loss_fn=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 5e-3},
    epochs=5,
    batch_size=32,
    with_weight_init=True,
    device="cuda",
)

model_phase1 = tracker.wrap_fit(
    model=model_phase1,
    dataset_val=galaxymnist_val,
    dataset_test=galaxymnist_test,
    metrics=metrics,
    log_model=False,
    run_name="phase_1_torch",
)
# Freeze base model, train only custom head
model_phase1.unfreeze_layers("all")
model_phase1.freeze_layers(["resnet50"])

print("\nPhase 1 layer trainability:")
for name, param in model_phase1.model_.named_parameters():
    print(f"Layer: {name}, Trainable: {param.requires_grad}")
print("\nStarting Phase 1 training...")
model_phase1.fit(
    dataset=galaxymnist_train,
    dataset_val=galaxymnist_val,
    metrics=metrics,
    fine_tune=True,
)

print("\n" + "=" * 60)
print("PHASE 1 COMPLETED")
print("=" * 60)

# ============================================================
# PHASE 2: FINE-TUNE LAYER4 BLOCK OF RESNET50
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2: Fine-tuning ResNet50 layer4 block")
print("Custom head + layer4 block trainable, rest frozen")
print("=" * 60 + "\n")

model_phase2 = PytorchModel(
    model_class=model_phase1.model_,
    model_params={},
    loss_fn=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 5e-4},
    epochs=15,
    batch_size=32,
    with_weight_init=True,
)

model_phase2 = tracker.wrap_fit(
    model_phase2,
    dataset_val=galaxymnist_val,
    dataset_test=galaxymnist_test,
    metrics=metrics,
    log_model=False,
    run_name="phase_2_torch",
)
# Unfreeze all layers first
model_phase2.unfreeze_layers("all")

# Freeze all ResNet50 layers except layer4 (equivalent to conv5 in Keras)
for name, param in model_phase2.model_.named_parameters():
    if name.startswith("resnet50"):
        # Check if it's NOT in layer4 (the last block before avgpool)
        if "resnet50.7" not in name:  # Layer 7 is layer4 in ResNet50
            param.requires_grad = False

print("\nPhase 2 layer trainability (sample):")
count = 0
for name, param in model_phase2.model_.named_parameters():
    if count < 10 or "resnet50.7" in name or "classifier" in name:
        print(f"Layer: {name}, Trainable: {param.requires_grad}")
    count += 1

print("\nStarting Phase 2 training...")
model_phase2.fit(
    dataset=galaxymnist_train,
    dataset_val=galaxymnist_val,
    metrics=metrics,
    fine_tune=True,
)

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 COMPLETED - Final Evaluation")
print("=" * 60 + "\n")
phase2_metrics = model_phase2.get_metrics(dataset=galaxymnist_test, metrics=metrics)
print(f"Phase 2 metrics: {phase2_metrics}")
