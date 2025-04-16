import os
from astrodata.data.schemas import RawData, ProcessedData


def extract_format(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()
    return {".fits": "fits", ".h5": "hdf5", ".csv": "csv", ".parquet": "parquet"}.get(
        ext, "unknown"
    )


def convert_to_processed_data(data: RawData) -> ProcessedData:
    """
    Convert RawData to ProcessedData.
    """
    features = data.data.iloc[:, :-1].values  # Extract all columns except the last
    labels = data.data.iloc[:, -1].values  # Extract the last column
    return ProcessedData(
        features=features,
        labels=labels,
        metadata={
            "source": data.source,
            "format": data.format,
        },
    )
