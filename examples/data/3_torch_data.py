from astrodata.data.loaders import TorchDataLoaderWrapper, TorchImageLoader

if __name__ == "__main__":
    loader = TorchImageLoader()
    print("TorchImageLoader initialized.")

    # Load the image data from the specified directory structure.
    # The directory should contain train/val/test folders with class subdirectories.

    raw_data = loader.load(
        "../../testdata/torch/mnist/"
    )  # Points to folder with train/val/test

    print("Image data loaded from directory structure.")
    print(f"train data class_to_idx: {raw_data.get_dataset('train').class_to_idx}")
    print(f"test data class_to_idx: {raw_data.get_dataset('test').class_to_idx}")
    # print(f"val data class_to_idx: {raw_data.get_dataset('val').class_to_idx}")

    # Define the DataLoader wrapper with desired settings for training.
    # This wrapper will create PyTorch DataLoaders
    dataloader_wrapper = TorchDataLoaderWrapper(
        batch_size=32,
        num_workers=0,
        pin_memory=False,
    )
    print("TorchDataLoaderWrapper initialized with training configuration.")

    # Create the actual PyTorch DataLoaders from the raw data.
    dataloaders = dataloader_wrapper.create_dataloaders(raw_data)
    print("PyTorch DataLoaders created successfully.")

    # Extract individual DataLoaders for each data split.
    # These DataLoaders can be directly used in PyTorch training loops.
    train_dataloader = dataloaders.get_dataloader("train")
    # val_dataloader = dataloaders.get_dataloader("val")
    test_dataloader = dataloaders.get_dataloader("test")

    # Show in detail what's inside the train DataLoader
    print("--" * 30)
    print("Train DataLoader details:")
    print(f"Number of batches in train DataLoader: {len(train_dataloader)}")
    print(f"Batch size: {train_dataloader.batch_size}")
    print("--" * 30)

    # Show how a train_dataloader can be accessed in a training loop
    for images, labels in train_dataloader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of labels shape: {labels.shape}")
        break
