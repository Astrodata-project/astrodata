import os
import tarfile

import pandas as pd
import requests
from astropy.io import fits


def download_and_load_fits(target_folder: str = "testdata") -> pd.DataFrame:
    url = "https://d12o80ict8oyer.cloudfront.net/catania_cavuoti.fit"
    file_name = f"{target_folder}/catania_cavuoti.fit"
    if not os.path.exists(file_name):
        response = requests.get(url)
        with open(file_name, "wb") as file:
            file.write(response.content)

    # Read FITS file
    with fits.open(file_name) as hdul:
        data = hdul[1].data
        df = pd.DataFrame(data.tolist(), columns=data.names)
    return df


def download_galaxy_mnist(target_folder: str = "testdata") -> str:
    url = "https://d12o80ict8oyer.cloudfront.net/cifar_galaxy_mnist.tar.gz"
    file_name = f"{target_folder}/cifar_galaxy_mnist.tar.gz"

    if not os.path.exists(file_name):
        response = requests.get(url)
        with open(file_name, "wb") as file:
            file.write(response.content)

    # Extract tar.gz file

    if not os.path.exists(f"{target_folder}/sorted_galaxy_mnist"):
        with tarfile.open(file_name, "r:gz") as tar:
            tar.extractall(path=f"{target_folder}")

    return f"{target_folder}/sorted_galaxy_mnist"
