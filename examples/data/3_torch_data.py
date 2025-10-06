import os

from torchvision import datasets

from astrodata.data.loaders import TorchDataLoaderWrapper, TorchLoader


# Helper function to save images
def save_dataset(out_dir, dataset, subset_name):
    subset_dir = os.path.join(out_dir, subset_name)
    os.makedirs(subset_dir, exist_ok=True)
    for idx, (img, label) in enumerate(dataset):
        # img is a PIL Image by default (mode L)
        label_dir = os.path.join(subset_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)
        # choose filename
        filename = os.path.join(label_dir, f"{idx}.png")
        img.save(filename)


def setup_datasets():
    data_dir = "../../testdata/torch/cifar10"
    os.makedirs(data_dir, exist_ok=True)

    cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
    cifar_test = datasets.CIFAR10(root=data_dir, train=False, download=True)

    save_dataset(data_dir, cifar_train, "train")
    save_dataset(data_dir, cifar_test, "test")


if __name__ == "__main__":
    loader = TorchLoader()
    print("TorchImageLoader initialized.")

    # Load the image data from the specified directory structure.
    # The directory should contain train/val/test folders with class subdirectories.

    # Setup: Download CIFAR10 and a dummy FITS dataset to the specified directory for testing
    setup_datasets()
    cifar_dir = "../../testdata/torch/cifar10"
    fits_dir = "../../testdata/torch/fits"

    cifar_data = loader.load(cifar_dir)  # Points to folder with train/val/test
    fits_data = loader.load(fits_dir)  # Points to folder with train/val/test

    print("Image data loaded from directory structure.")
    print(
        f"Cifar train data class_to_idx: {cifar_data.get_dataset('train').class_to_idx}"
    )
    print(
        f"Cifar test data class_to_idx: {cifar_data.get_dataset('test').class_to_idx}"
    )
    # print(f"val data class_to_idx: {cifar_data.get_dataset('val').class_to_idx}")

    # Define the DataLoader wrapper with desired settings for training.
    # This wrapper will create PyTorch DataLoaders
    dataloader_wrapper = TorchDataLoaderWrapper(
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )
    print("TorchDataLoaderWrapper initialized with training configuration.")

    # Create the actual PyTorch DataLoaders from the raw data.
    cifar_dataloaders = dataloader_wrapper.create_dataloaders(cifar_data)
    print("PyTorch DataLoaders created successfully.")

    # Extract individual DataLoaders for each data split.
    # These DataLoaders can be directly used in PyTorch training loops.
    train_dataloader = cifar_dataloaders.get_dataloader("train")
    # val_dataloader = dataloaders.get_dataloader("val")
    test_dataloader = cifar_dataloaders.get_dataloader("test")

    # Show in detail what's inside the train DataLoader
    print("--" * 30)
    print("Cifar DataLoader details:")
    print("Train DataLoader details:")
    print(f"Number of batches in train DataLoader: {len(train_dataloader)}")
    print(f"Batch size: {train_dataloader.batch_size}")
    print("--" * 30)

    # Show how a train_dataloader can be accessed in a training loop
    for images, labels in train_dataloader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break

    train_dataloader = fits_data.get_dataset("train")
    test_dataloader = fits_data.get_dataset("test")

    # Below is to show the FITS dataloader details
    print("--" * 30)
    print("FITS DataLoader details:")
    print("Train DataLoader details:")
    print(f"Number of samples in train DataLoader: {len(train_dataloader)}")
    print(f"Batch size: {train_dataloader.batch_size}")
    print("--" * 30)
    # Show how a train_dataloader can be accessed in a training loop
    for image, label in train_dataloader:
        print(f"Image tensor shape: {image.shape}")
        print(f"Label: {label}")
        break
