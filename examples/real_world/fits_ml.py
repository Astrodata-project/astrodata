import hyperopt.hp as hp
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from astropy.io import fits

from astrodata.data import BaseLoader, DataPipeline, AbstractProcessor, RawData
from astrodata.preml import PremlPipeline, TrainTestSplitter
from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.model_selection import HyperOptSelector
from astrodata.ml.models import SklearnModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker
from testdata import download_fits

file_path = download_fits()
SEED = 42
config_path = "./config.yaml"


# First, let's define a custom loader for FITS files
# Astrodata provides loaders for common formats like CSV and Parquet,
# but FITS files require a custom implementation.
class FitsLoader(BaseLoader):
    def load(self, path: str):
        with fits.open(path) as hdul:
            data = hdul[1].data
            df = pd.DataFrame(data.tolist(), columns=data.names)
        return RawData(source=path, format="fits", data=df)


# Then, we define a custom processor to handle FITS data
# This processor will filter out invalid data and prepare the dataset for modeling.
class FitsProcessor(AbstractProcessor):
    def process(self, raw):
        objin = raw.data.shape[0]
        # Let's ingore some columns, identify the target columns, use the rest as features
        ignore = ["specObjID", "objid", "ra", "dec", "targetObjID", "zErr"]
        features = [col for col in raw.data.columns if col not in ignore]
        # Filter out negative values in features
        raw.data = raw.data[(raw.data[features] >= 0).all(axis=1)]
        objout = raw.data.shape[0]
        print("from ", objin, "intial objects we have now", objout)
        print("object discarded:", objin - objout)
        raw.data = raw.data.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
        return raw


loader = FitsLoader()
processor = FitsProcessor()

# Define the data pipeline with the config file, loader and processors.
data_pipeline = DataPipeline(
    config_path=config_path,
    loader=loader,
    processors=[processor],
)
# Run the data pipeline to load and process the FITS data.
# This results in a ProcessedData object, ready for further preprocessing.
data = data_pipeline.run(file_path, dump_output=False)

tts = TrainTestSplitter(
    targets=["z"],
    train_size=0.2,
    random_state=SEED,
)
# Define the PremlPipeline with the TrainTestSplitter processor.
preml_pipeline = PremlPipeline(
    config_path=config_path,
    processors=[tts],
)

# Run the PremlPipeline to split the data into training and testing sets.
preml_data = preml_pipeline.run(data, dump_output=False)

X_train, X_test, y_train, y_test = preml_data.dump_supervised_ML_format()
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


tracker = SklearnMLflowTracker(
    run_name="catania_cavuoti_hyperopt",
    experiment_name="catania_cavuoti_hyperopt",
    extra_tags=None,
)

# Define the metrics to be used for evaluation

mse = SklearnMetric(mean_squared_error, greater_is_better=False)
r2score = SklearnMetric(r2_score, greater_is_better=True)
mae = SklearnMetric(mean_absolute_error, greater_is_better=False)

metrics = [mse, r2score, mae]

rfr = SklearnModel(model_class=RandomForestRegressor)
gbr = SklearnModel(model_class=GradientBoostingRegressor)
svr = SklearnModel(model_class=SVR)

models = [rfr, gbr, svr]

param_space = {
    "model": hp.choice("model", models),
}

hos = HyperOptSelector(
    param_space=param_space,
    scorer=r2score,
    use_cv=False,
    random_state=42,
    max_evals=10,
    metrics=None,
    tracker=tracker,
)

hos.fit(X=X_train, y=y_train, X_val=X_test, y_val=y_test)

print("Best parameters found: ", hos.get_best_params())
print("Best metrics: ", hos.get_best_metrics())


tracker.register_best_model(
    metric=r2score,
    split_name="val",
    stage="Production",
)
