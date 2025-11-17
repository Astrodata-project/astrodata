import tensorflow as tf

from astrodata.data.loaders.tensorflow_loader import TensorflowLoader


def describe_dataset(name, ds):
    card = tf.data.experimental.cardinality(ds).numpy()
    print("--" * 30)
    print(f"{name} tf.data.Dataset details:")
    print(f"Cardinality (batches): {card if card >= 0 else 'unknown'}")

    # Peek one batch
    for images, labels in ds.take(1):
        print(f"[{name}] Example batch images tensor shape: {images.shape}")
        print(f"[{name}] Example batch labels tensor shape: {labels.shape}")
        break
    print("--" * 30)


if __name__ == "__main__":
    cifar_dir = "../../testdata/torch/cifar10"
    fits_dir = "../../testdata/torch/fits"

    # Initialize loader
    loader = TensorflowLoader()
    print("KerasLoader initialized.")

    print("Loading CIFAR10 directory-structured dataset with Keras...")
    cifar_data = loader.load(cifar_dir, image_size=(32, 32))
    print("Loaded CIFAR10 KerasData.")

    # Class mappings (from train split)
    print(f"CIFAR10 class_names: {cifar_data.metadata.get('class_names')}")
    print(f"CIFAR10 class_to_idx: {cifar_data.metadata.get('class_to_idx')}")

    cifar_train = cifar_data.get_dataset("train")
    cifar_test = cifar_data.get_dataset("test")

    describe_dataset("CIFAR10 Train", cifar_train)
    describe_dataset("CIFAR10 Test", cifar_test)

    # Demonstrate iterating a single batch from the train dataset
    for images, labels in cifar_train.take(1):
        print(f"[CIFAR10] Batch example images shape: {images.shape}")
        print(f"[CIFAR10] Batch example labels shape: {labels.shape}")
        break

    print("Loading FITS directory-structured dataset with Keras...")
    fits_data = loader.load(fits_dir, batch_size=1)

    print("Loaded FITS KerasData.")

    # class mappings (from train split)
    print(f"FITS class_names: {fits_data.metadata.get('class_names')}")
    print(f"FITS class_to_idx: {fits_data.metadata.get('class_to_idx')}")

    fits_train = fits_data.get_dataset("train")
    fits_test = fits_data.get_dataset("test")

    describe_dataset("FITS Train", fits_train)
    describe_dataset("FITS Test", fits_test)

    for images, labels in fits_train.take(1):
        print(f"[FITS] Batch example images shape: {images.shape}")
        print(f"[FITS] Batch example labels shape: {labels.shape}")
        break
