import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from astropy.io import fits

from astrodata.data.schemas import ProcessedData, RawData

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
FITS_EXTS = {".fits", ".fit", ".fts"}


def extract_format(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()
    return {
        ".fits": "fits",
        ".hdf5": "hdf5",
        ".csv": "csv",
        ".parquet": "parquet",
    }.get(ext, "unknown")


def convert_to_processed_data(data: RawData) -> ProcessedData:
    """
    Convert RawData to ProcessedData.

    Parameters:
        data: RawData instance containing data and metadata.

    Returns:
        ProcessedData with the same data payload and a metadata dict containing
        "source" and "format" copied from the RawData.
    """

    return ProcessedData(
        data=data.data,
        metadata={
            "source": data.source,
            "format": data.format,
        },
    )


def list_class_dirs(root: Path) -> List[Path]:
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    class_dirs.sort()
    return class_dirs


def build_class_index(class_dirs: List[Path]) -> Tuple[List[str], Dict[str, int]]:
    class_names = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return class_names, class_to_idx


def gather_paths_and_labels(
    image_dir: Path,
    valid_exts: Optional[Iterable[str]] = None,
    return_type: str = "str",  # "str" | "path"
) -> Tuple[List, List[int], List[str], Dict[str, int]]:
    """
    Traverse class directories and collect file paths and labels.

    Assumes a structure:
      image_dir/
        class_a/
          file1...
        class_b/
          file2...

    Parameters:
        image_dir: Root directory containing class subdirectories.
        valid_exts: Optional iterable of allowed file extensions (lowercased, including leading dot).
        return_type: "str" to return string paths, "path" to return Path objects.

    Returns:
        - paths: List of file paths (str or Path based on return_type).
        - labels: List of integer class indices aligned with paths.
        - class_names: List of class names.
        - class_to_idx: Mapping from class name to index.

    Notes:
        Files are sorted per class directory; classes are sorted by name.
    """
    class_dirs = list_class_dirs(image_dir)
    class_names, class_to_idx = build_class_index(class_dirs)

    paths = []
    labels = []

    ext_set = set(valid_exts) if valid_exts is not None else None

    for d in class_dirs:
        files = [p for p in d.iterdir() if p.is_file()]
        files.sort()
        for f in files:
            if ext_set is not None and f.suffix.lower() not in ext_set:
                continue
            paths.append(str(f) if return_type == "str" else f)
            labels.append(class_to_idx[d.name])

    return paths, labels, class_names, class_to_idx


def decode_fits(path: str) -> np.ndarray:
    """
    Load image data from a FITS file as a float32 NumPy array in HWC layout.

    Shapes handled:
        - 2D: [H, W] -> returns [H, W, 1]
        - 3D: [C, H, W] (C <= 4) -> returns [H, W, C]
        - 3D: [H, W, C] (C <= 4) -> returns as-is

    Parameters:
        path: Path to a FITS file.

    Returns:
        NumPy array of shape [H, W, C] with dtype float32.

    Raises:
        ValueError: If no image data is found or shape is unsupported.
    """
    with fits.open(path) as hdul:
        hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
        if hdu is None or hdu.data is None:
            raise ValueError(f"No image data found in FITS file: {path}")

        data = np.array(hdu.data, dtype=np.float32, copy=True)

        if data.ndim == 2:
            data = np.expand_dims(data, -1)  # H, W -> H, W, 1
        elif data.ndim == 3:
            if data.shape[0] <= 4:
                data = np.moveaxis(data, 0, -1)  # C,H,W -> H,W,C
            elif data.shape[-1] <= 4:
                pass  # already H,W,C
            else:
                raise ValueError(
                    f"3D FITS data does not look like multi-channel image (shape {data.shape}) in {path}"
                )
        else:
            raise ValueError(
                f"Expected 2D or 3D FITS image, got shape {data.shape} in {path}"
            )

        return data
