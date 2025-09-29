from astrodata.data import AbstractProcessor, CsvLoader, DataPipeline, RawData


def dummy_csv_file():
    data = "feature1,feature2,target\n1,5,10\n2,4,20\n3,3,30\n4,2,40\n5,1,50"
    with open("dummy_data.csv", "w") as f:
        f.write(data)
    return "dummy_data.csv"


def remove_dummy_csv_file():
    import os

    if os.path.exists("dummy_data.csv"):
        os.remove("dummy_data.csv")


if __name__ == "__main__":
    # This example demonstrates how to use the DataPipeline along with AbstractProcessor and BaseLoader classes to process data.
    # The DataPipeline class orchestrates the loading and processing of data through a series of defined processors.
    # It should be used when you want to apply a sequence of transformations to your data, aimed to prepare it for machine learning tasks.
    # We will create a dummy CSV file, load it using CsvLoader, and process it with a custom processor.

    # Create a dummy CSV file for demonstration purposes
    data_path = dummy_csv_file()
    print(f"Dummy CSV file created at: {data_path}")

    # Define the loader that will be used to load the data, returning a RawData object. This object standardizes the data format to a pandas DataFrame.
    # This is the format which will be used by the processors in the pipeline.
    loader = CsvLoader()
    print("CsvLoader initialized.")

    # Define a list of processors to be applied to the data. In this case, we will use a custom processor that creates a third feature.
    # Our custom processor needs to inherit from AbstractProcessor and implement the process method.
    class CustomProcessor(AbstractProcessor):
        def process(self, raw: RawData) -> RawData:
            raw.data["feature3"] = raw.data["feature1"] + raw.data["feature2"]
            return raw

    print("CustomProcessor defined.")

    # Define the list of processors to be used in the pipeline.
    # The processors will be applied in the order they are defined.
    data_processors = [CustomProcessor()]
    print("Data processors defined.")

    # Define the data pipeline with the loader and processors.
    config_path = "example_config.yaml"
    data_pipeline = DataPipeline(
        config_path=config_path, loader=loader, processors=data_processors
    )
    print(
        f"DataPipeline initialized with {loader.__class__.__name__} and {[p.__class__.__name__ for p in data_processors]}."
    )

    # Run the data pipeline with the path to the dummy CSV file.
    # This will load the data, apply the processors, and return a ProcessedData object, which contains the processed data in a pandas DataFrame.

    processed_data = data_pipeline.run(data_path, dump_output=False)

    print("Data Pipeline ran successfully!")
    print(f"Processed data head:\n {processed_data.data.head()}")
    print(f"Processed data shape: {processed_data.data.shape}")
    print(f"Processed data metadata: {processed_data.metadata}")
    remove_dummy_csv_file()
