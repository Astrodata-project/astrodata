import os
from pathlib import Path

from astropy.utils.data import download_file
from torchvision import datasets

from astrodata.data.loaders import TorchDataLoaderWrapper, TorchLoader


def save_dataset(out_dir, dataset, subset_name):
    subset_dir = out_dir / subset_name
    if subset_dir.exists():
        return
    subset_dir.mkdir(parents=True, exist_ok=True)
    for idx, (img, label) in enumerate(dataset):
        label_dir = subset_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        filename = label_dir / f"{idx}.png"
        if not filename.exists():
            img.save(filename)


def setup_datasets(cifar_dir, fits_dir):
    cifar_path = Path(cifar_dir)
    fits_path = Path(fits_dir)

    cifar_path.mkdir(parents=True, exist_ok=True)
    fits_path.mkdir(parents=True, exist_ok=True)

    cifar_train = datasets.CIFAR10(root=str(cifar_path), train=True, download=True)
    cifar_test = datasets.CIFAR10(root=str(cifar_path), train=False, download=True)

    split_ds_map = {"train": cifar_train, "test": cifar_test}
    for split in ("train", "test"):
        ds = split_ds_map.get(split)
        if ds is not None:
            save_dataset(cifar_path, ds, split)

    image_file = download_file(
        "http://data.astropy.org/tutorials/FITS-images/HorseHead.fits", cache=True
    )

    with open(image_file, "rb") as f:
        image_data = f.read()
    for split in ("train", "test"):
        for clss in ("first", "second"):
            cls_dir = fits_path / split / clss
            cls_dir.mkdir(parents=True, exist_ok=True)
            target_file = cls_dir / "image.fits"
            if not target_file.exists():
                with open(target_file, "wb") as out_f:
                    out_f.write(image_data)


if __name__ == "__main__":

    # ==============================================================
    # DATA SETUP (executed once for both classic image + FITS examples)
    # This will:
    # 1. Download CIFAR10.
    # 2. Materialize its train/test splits into an ImageFolder-like directory tree.
    # 3. Download a single example FITS file and duplicate it into a dummy
    #    train/test, multi-class directory tree for demonstration.
    # ==============================================================
    print("Setting up datasets...")
    cifar_dir = "../../testdata/torch/cifar10"
    fits_dir = "../../testdata/torch/fits"
    setup_datasets(cifar_dir, fits_dir)

    # Initialize the TorchLoader, which can handle both classic images and FITS
    loader = TorchLoader()
    print("TorchImageLoader initialized.")

    # ==============================================================
    # SECTION 1: CLASSIC (RGB) IMAGE DATA EXAMPLE (CIFAR10)
    # --------------------------------------------------------------
    # Goal: Show how a standard natural-image style dataset (already in a
    #       folder layout with class subdirectories) is loaded and wrapped
    #       into PyTorch DataLoaders.
    # ==============================================================

    # Load directory-structured CIFAR10 dataset (train/test folders)
    cifar_data = loader.load(cifar_dir)
    print("Loaded CIFAR10 structured dataset object.")

    # Display class mappings for both splits
    print(f"CIFAR10 train class_to_idx: {cifar_data.get_dataset('train').class_to_idx}")
    print(f"CIFAR10 test  class_to_idx: {cifar_data.get_dataset('test').class_to_idx}")

    # Create DataLoader wrapper (shared config, could be tuned per section)
    dataloader_wrapper = TorchDataLoaderWrapper(
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )
    print("Initialized TorchDataLoaderWrapper.")

    # Build actual PyTorch DataLoaders
    cifar_dataloaders = dataloader_wrapper.create_dataloaders(cifar_data)
    print("Created CIFAR10 DataLoaders.")

    # Extract split-specific loaders
    cifar_train_loader = cifar_dataloaders.get_dataloader("train")
    cifar_test_loader = cifar_dataloaders.get_dataloader("test")

    # Introspect the train DataLoader
    print("--" * 30)
    print("CIFAR10 Train DataLoader details:")
    print(f"Number of batches: {len(cifar_train_loader)}")
    print(f"Batch size: {cifar_train_loader.batch_size}")
    print("--" * 30)

    # Demonstrate single training batch access
    for images, labels in cifar_train_loader:
        print(f"[CIFAR10] Example batch images tensor shape: {images.shape}")
        print(f"[CIFAR10] Example batch labels tensor shape: {labels.shape}")
        break

    # ==============================================================
    # SECTION 2: FITS (ASTRONOMICAL) IMAGE DATA EXAMPLE
    # --------------------------------------------------------------
    # Goal: Show how the same loader abstraction can handle FITS images
    #       arranged in the same folder pattern:
    #         fits_dir/
    #             train/first/*.fits
    #             train/second/*.fits
    #             test/first/*.fits
    #             test/second/*.fits
    #
    # The FITS setup duplicated one sample file to keep this extremely small.
    # We still wrap it in DataLoaders to mimic a real workflow.
    # ==============================================================

    fits_data = loader.load(fits_dir)
    print("Loaded FITS structured dataset object.")
    print(f"FITS train class_to_idx: {fits_data.get_dataset('train').class_to_idx}")
    print(f"FITS test  class_to_idx: {fits_data.get_dataset('test').class_to_idx}")

    # Reuse the same wrapper (could alternatively instantiate with different params)
    fits_dataloaders = dataloader_wrapper.create_dataloaders(fits_data)
    print("Created FITS DataLoaders.")

    fits_train_loader = fits_dataloaders.get_dataloader("train")
    fits_test_loader = fits_dataloaders.get_dataloader("test")

    print("--" * 30)
    print("FITS Train DataLoader details:")
    print(f"Number of batches: {len(fits_train_loader)}")
    print(f"Batch size: {fits_train_loader.batch_size}")
    print("--" * 30)

    # Show a single batch (each element a single-channel tensor)
    for images, labels in fits_train_loader:
        print(f"[FITS] Example batch images tensor shape: {images.shape}")
        print(f"[FITS] Example batch labels tensor shape: {labels.shape}")
        break
