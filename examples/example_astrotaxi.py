from astrodata.data import (
    ParquetLoader,
    RawData,
    AbstractPreprocessor,
    MissingValueImputer,
    DropDuplicates,
    DataPipeline,
)
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# define loader
loader = ParquetLoader()


preprocessors = [MissingValueImputer(), DropDuplicates()]

pipeline = DataPipeline(loader, preprocessors)

data_path = "./testdata/green_tripdata_2024-01.parquet"

processed = pipeline.run(data_path)

print("Pipeline ran successfully!")
print(f"Processed data shape:{processed.data.shape}")
