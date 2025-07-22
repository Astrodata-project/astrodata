from astrodata.data import AbstractProcessor, DataPipeline, ParquetLoader, RawData


def run_data_import_example(config, tracker):
    # This step  demonstrates how to use the DataPipeline with a ParquetLoader and a custom processor.
    # The DataPipeline class orchestrates the loading and processing of data through a series of defined processors.
    # It should be used when you want to apply a sequence of transformations to your data, aimed to prepare it for machine learning tasks.
    # We will load a Parquet file, process it with a custom processor, and track the resulting data.

    # Define the loader that will be used to load the data, returning a RawData object.
    loader = ParquetLoader()

    # Define a custom processor to create a target variable and filter the data.
    # The processor needs to inherit from AbstractProcessor and implement the process method.
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

    # Define the list of processors to be used in the pipeline.
    data_processors = [TargetCreator()]

    # Define the data pipeline with the config file, loader and processors.
    data_pipeline = DataPipeline(
        config_path=config, loader=loader, processors=data_processors
    )

    # Path to the input Parquet file
    data_path = "./testdata/green_tripdata_2024-01.parquet"

    # Run the data pipeline with the path to the Parquet file.
    processed = data_pipeline.run(data_path)

    # The tracker is used to version code and data with Git and DVC.
    # The track method will version everything that is included in the config file, alongside with astrodat-produced files.
    # Astrodata creates a folder named "astrodata_files" in which it stores generated data and artifacts.
    tracker.track("Data pipeline run, processed data versioned")

    print("Data Pipeline ran successfully!")
    print(f"Processed data shape:{processed.data.shape}")

    return processed
