# AstroData ML Toolkit

AstroData is a modular Python toolkit for data processing, machine learning, and experiment tracking, designed for astrophysical data workflows. This project is part of Italy’s National Recovery and Resilience Plan (PNRR), coordinated by the National Institute for Astrophysics (INAF) under the Spoke 3 initiative, which focuses on advanced data analysis and artificial intelligence for astrophysics.

It provides robust pipelines for data ingestion, preprocessing, model training, evaluation, and experiment tracking with integrated support for Git and DVC.

## Features

- **Data Pipelines**: Modular data loaders, processors, and schemas for handling raw and processed data.
- **Preprocessing**: Utilities for normalization, one-hot encoding, and custom preprocessing steps.
- **Machine Learning**: Abstract base classes and utilities for metrics, model selection, and model management.
- **Experiment Tracking**: Integrated code and data tracking using Git and DVC, with MLflow support.
- **Extensible**: Easily extend with custom data processors, models, and metrics.

## Installation

Install the package directly from GitHub using any package manager such as pip, uv, and conda.
```sh
pip install git+https://github.com/Astrodata-project/astrodata.git
```

## Project Structure

```
astrodata/
    data/         # Data loaders, schemas, processors, and utilities
    ml/           # ML metrics, model selection, and models
    preml/        # Preprocessing pipelines and processors
    tracking/     # Code/data tracking (Git, DVC, MLflow)
    utils/        # Logging and general utilities
docs/             # Sphinx documentation
examples/         # Example scripts and notebooks
tests/            # Unit and integration tests
```

## Usage

### Data Pipeline Example

```python
data_path = dummy_csv_file()

loader = CsvLoader()

class CustomProcessor(AbstractProcessor):
    def process(self, raw: RawData) -> RawData:
        raw.data["feature3"] = raw.data["feature1"] + raw.data["feature2"]
        return raw

data_processors = [CustomProcessor()]

data_pipeline = DataPipeline(loader, data_processors)

processed_data = data_pipeline.run(data_path)

```

### Model Training Example

```python
model = SklearnModel(model_class=LinearSVR, random_state=42)

mae = SklearnMetric(mean_absolute_error, greater_is_better=False)
mse = SklearnMetric(mean_squared_error)
r2 = SklearnMetric(r2_score, greater_is_better=True)
msle = SklearnMetric(mean_squared_log_error)

metrics = [mae, mse, r2, msle]


model.fit(X_train, y_train)

metrics = model.get_metrics(
    X_test,
    y_test,
    metrics=metrics,
)
```

### Experiment Tracking

```python
gradientboost = SklearnModel(model_class=GradientBoostingClassifier)

tracker = SklearnMLflowTracker(
    run_name="MlFlowSimpleRun",
    experiment_name="examples_ml_5_mlflow_simple_example.py",
    extra_tags={"stage": "testing"},
)

accuracy = SklearnMetric(accuracy_score)
f1 = SklearnMetric(f1_score, average="micro")
logloss = SklearnMetric(log_loss, greater_is_better=False)

metrics = [accuracy, f1, logloss]

tracked_gradientboost = tracker.wrap_fit(
    gradientboost, X_test=X_test, y_test=y_test, metrics=metrics, log_model=True
)

tracked_gradientboost.fit(X_train, y_train)
```

## Documentation

Documentation is available in the [project's GitHub.io pages](https://astrodata-project.github.io/astrodata/).

## Contributing

Contributions are welcome! Please open issues or pull requests.

## License

See [LICENSE](LICENSE) for details.