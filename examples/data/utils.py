from pathlib import Path

from astropy.utils.data import download_file
from torchvision import datasets


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
