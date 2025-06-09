from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from xgboost import XGBRegressor

from astrodata.data import AbstractProcessor, DataPipeline, ParquetLoader, RawData
from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.GridSearchSelector import GridSearchCVSelector
from astrodata.ml.models.XGBoostModel import XGBoostModel
from astrodata.preml import OHE, MissingImputator, PremlPipeline
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker

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

ohe_processor = OHE(
    # categorical_columns=["PULocationID", "DOLocationID"],
    # numerical_columns=["passenger_count", "trip_distance", "duration"],
    categorical_columns=["PULocationID"],
    numerical_columns=["trip_distance"],
    save_path="./testdata/ohe.pkl",
)

missing_imputator = MissingImputator(
    categorical_columns=["PULocationID"],
    numerical_columns=["trip_distance"],
    save_path="./testdata/imputer.pkl",
)

preml_pipeline = PremlPipeline([missing_imputator, ohe_processor], config_path)

preml_data = preml_pipeline.run(processed)

print("Preml Pipeline ran successfully!")
print(f"Preml data shape:{preml_data.train_features.shape}")
print(f"Preml data shape:{preml_data.train_targets.shape}")

X_train, X_test, y_train, y_test = preml_data.dump_supervised_ML_format()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Instantiate and configure the XGBoost model
gradientboost = XGBoostModel(model_class=XGBRegressor)

tracker = SklearnMLflowTracker(
    log_model=True,
    run_name="DemoRun",
    experiment_name="examples/example_astrotaxi.py",
    extra_tags={"stage": "testing"},
)

metrics = [
    SklearnMetric(root_mean_squared_error),
    SklearnMetric(mean_absolute_error),
    SklearnMetric(mean_squared_error),
    SklearnMetric(r2_score),
]

tracked_gradientboost = tracker.wrap_fit(
    model=gradientboost,
    X_test=X_test,
    y_test=y_test,
    input_example=X_train.iloc[:5],
    metrics=metrics,
)

gss = GridSearchCVSelector(
    tracked_gradientboost,
    param_grid={
        "n_estimators": [50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5],
    },
    n_jobs=6,
)

gss.fit(X_train, y_train)
