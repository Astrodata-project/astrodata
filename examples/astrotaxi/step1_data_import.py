from astrodata.data import AbstractProcessor, DataPipeline, ParquetLoader, RawData


def run_data_import_example(config, tracker):
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

    data_pipeline = DataPipeline(
        config_path=config, loader=loader, processors=data_processors
    )

    data_path = "./testdata/green_tripdata_2024-01.parquet"

    processed = data_pipeline.run(data_path)
    tracker.track("Data pipeline run, processed data versioned")

    print("Data Pipeline ran successfully!")
    print(f"Processed data shape:{processed.data.shape}")

    return processed
