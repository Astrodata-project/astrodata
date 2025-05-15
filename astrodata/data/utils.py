import os
from astrodata.data.schemas import RawData, ProcessedData


def extract_format(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()
    return {".fits": "fits", ".h5": "hdf5", ".csv": "csv", ".parquet": "parquet"}.get(
        ext, "unknown"
    )


def convert_to_processed_data(data: RawData) -> ProcessedData:
    """
    Convert RawData to ProcessedData using specified feature and target columns.
    """

    return ProcessedData(
        data=data.data,
        metadata={
            "source": data.source,
            "format": data.format,
        },
    )
