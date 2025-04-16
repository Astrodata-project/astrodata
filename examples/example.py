from astrodata.data.loaders.parquet_loader import ParquetLoader
from astrodata.data.preprocessors.base import (
    MissingValueImputer,
    DropDuplicates,
)
from astrodata.data.pipeline import DataPipeline

loader = ParquetLoader()

preprocessors = [MissingValueImputer(), DropDuplicates()]

pipeline = DataPipeline(loader, preprocessors)

data_path = "./testdata/green_tripdata_2024-01.parquet"

processed = pipeline.run(data_path)

print("Pipeline ran successfully!")
print(f"Features shape:{processed.features.shape}")
print(f"Labels shape:{processed.labels.shape}")
