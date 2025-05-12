from astrodata.data import (
    RawData,
    AbstractProcessor,
    ParquetLoader,
    DataPipeline,
)
from astrodata.preml import PremlPipeline

# define loader
loader = ParquetLoader()


class TargetCreator(AbstractProcessor):
    def process(self, raw: RawData) -> RawData:
        raw.data["duration"] = (
            raw.data.lpep_dropoff_datetime - raw.data.lpep_pickup_datetime
        )
        raw.data["duration"] = raw.data["duration"].apply(
            lambda x: x.total_seconds() / 60
        )
        raw.data = raw.data[
            (raw.data["duration"] >= 1) & (raw.data["duration"] <= 60)
        ].reset_index(drop=True)
        raw.data = raw.data[raw.data["trip_distance"] < 50].reset_index(drop=True)
        return raw


data_processors = [TargetCreator()]

data_pipeline = DataPipeline(loader, data_processors)

data_path = "./testdata/green_tripdata_2024-01.parquet"

processed = data_pipeline.run(data_path)

print("Data Pipeline ran successfully!")
print(f"Processed data shape:{processed.data.shape}")

config_path = "./examples/example_config.yaml"

preml_pipeline = PremlPipeline([], config_path)

preml_data = preml_pipeline.run(processed)

print("Preml Pipeline ran successfully!")
print(f"Preml data shape:{preml_data.train_features.shape}")
print(f"Preml data shape:{preml_data.train_targets.shape}")
