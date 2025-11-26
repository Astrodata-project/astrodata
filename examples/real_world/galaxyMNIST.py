import keras
import tensorflow as tf
from keras.metrics import Accuracy, SparseCategoricalAccuracy

from astrodata.data.loaders.tensorflow_loader import TensorflowLoader
from astrodata.ml.metrics import TensorflowMetric
from astrodata.ml.models import TensorflowModel
from astrodata.tracking.MLFlowTracker import TensorflowMLflowTracker
from testdata import download_galaxy_mnist

if not tf.executing_eagerly():
    tf.config.run_functions_eagerly(True)
data_path = download_galaxy_mnist()


print(f"GalaxyMNIST data downloaded and extracted to: {data_path}")

img_size = (64, 64)

loader = TensorflowLoader()
cifar_data = loader.load(data_path, image_size=img_size)

print(f"GalaxyMNIST class_names: {cifar_data.metadata.get('class_names')}")
print(f"GalaxyMNIST class_to_idx: {cifar_data.metadata.get('class_to_idx')}")

galaxymnist_train = cifar_data.get_dataset("train")
galaxymnist_test = cifar_data.get_dataset("test")
galaxymnist_val = cifar_data.get_dataset("val")

# Get number of classes from the metadata
num_classes = len(cifar_data.metadata.get("class_names", []))
print(f"Number of classes: {num_classes}")

base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3),  # GalaxyMNIST image size
)

augment = keras.Sequential(
    [
        keras.layers.RandomFlip(mode="horizontal"),
        keras.layers.RandomRotation(0.2),
        keras.layers.RandomContrast(0.2),
    ],
    name="data_augmentation",
)

custom_head = keras.Sequential(
    [
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),  # Add dropout for regularization
        keras.layers.Dense(num_classes, activation="softmax", name="galaxy_classifier"),
    ],
    name="galaxy_mnist_head",
)

inputs = keras.Input(shape=(img_size[0], img_size[1], 3), name="input_layer")
x = augment(inputs)
x = keras.applications.resnet50.preprocess_input(x)
x = base_model(x, training=False)
x = custom_head(x)

galaxy_minst_resnet = keras.Model(inputs, x, name="ResNet50_GalaxyMNIST")

print(f"Model created with {num_classes} output classes")
print("Model summary:")
galaxy_minst_resnet.summary()

accuracy_metric = TensorflowMetric(
    SparseCategoricalAccuracy(), name="accuracy", greater_is_better=True
)
metrics = [accuracy_metric]

tracker = TensorflowMLflowTracker(
    run_name="GalaxyMNIST_ResNet50_FineTuning",
    experiment_name="GalaxyMNIST_ResNet50_FineTuning",
    extra_tags={"stage": "testing"},
)

model_phase1 = TensorflowModel(
    model_class=galaxy_minst_resnet,
    model_params={},
    loss_fn=keras.losses.SparseCategoricalCrossentropy,
    optimizer=keras.optimizers.Adam,
    optimizer_params={"learning_rate": 5e-3},
    epochs=5,
    batch_size=32,
    with_weight_init=True,
)

model_phase1 = tracker.wrap_fit(
    model=model_phase1,
    dataset_val=galaxymnist_val,
    dataset_test=galaxymnist_test,
    metrics=metrics,
    log_model=True,
)

model_phase1.unfreeze_layers("all")
model_phase1.freeze_layers(["input_layer", "data_augmentation", "resnet50"])

for layer in model_phase1.model_.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")


model_phase1.fit(
    dataset=galaxymnist_train,
    dataset_val=galaxymnist_val,
    metrics=metrics,
    fine_tune=True,
)

print("Phase 1 completed.")
print("Starting Phase 2 fine-tuning...")

model_phase2 = TensorflowModel(
    model_class=model_phase1.model_,
    model_params={},
    loss_fn=keras.losses.SparseCategoricalCrossentropy,
    optimizer=keras.optimizers.Adam,
    optimizer_params={"learning_rate": 5e-4},
    epochs=15,
    batch_size=32,
    with_weight_init=True,
)

model_phase2 = tracker.wrap_fit(
    model_phase2,
    dataset_val=galaxymnist_val,
    dataset_test=galaxymnist_test,
    metrics=metrics,
    log_model=True,
)

model_phase2.model_.layers
model_phase2.unfreeze_layers("all")
model_phase2.freeze_layers(["input_layer", "data_augmentation"])
tofreeze = []

for _layer in model_phase2.model_.layers[2].layers:
    if not _layer.name.startswith("conv5"):
        tofreeze.append(_layer.name)

model_phase2.freeze_layers(tofreeze, "resnet50")

for _layer in model_phase2.model_.layers[2].layers[0:10]:
    print(f"Layer: {_layer.name}, Trainable: {_layer.trainable}")

for layer in model_phase2.model_.layers:
    print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

model_phase2.fit(
    dataset=galaxymnist_train,
    dataset_val=galaxymnist_val,
    metrics=metrics,
    fine_tune=True,
)

# Evaluate Phase 2
print("Phase 2 evaluation:")
phase2_metrics = model_phase2.get_metrics(dataset=galaxymnist_test, metrics=metrics)
print(f"Phase 2 metrics: {phase2_metrics}")
