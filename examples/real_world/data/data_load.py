import os

import pandas as pd
import requests
from astropy.io import fits


def download_and_load_fits() -> pd.DataFrame:
    url = "https://d12o80ict8oyer.cloudfront.net/catania_cavuoti.fit"
    file_name = "examples/real_world/data/catania_cavuoti.fit"
    if not os.path.exists(file_name):
        response = requests.get(url)
        with open(file_name, "wb") as file:
            file.write(response.content)

    # Read FITS file
    with fits.open(file_name) as hdul:
        data = hdul[1].data
        df = pd.DataFrame(data.tolist(), columns=data.names)
    return df
