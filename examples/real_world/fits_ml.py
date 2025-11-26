import hyperopt.hp as hp
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.model_selection import HyperOptSelector
from astrodata.ml.models import SklearnModel
from astrodata.tracking.MLFlowTracker import SklearnMLflowTracker
from testdata import download_and_load_fits

df = download_and_load_fits()
SEED = 42

# Let's ingore some columns, identify the target columns, use the rest as features
ignore = ["specObjID", "objid", "ra", "dec", "targetObjID", "zErr"]
target = "z"
features = [col for col in df.columns if col not in ignore + [target]]


# Filter out negative values in features
objinthecatalog = df.shape[0]
df = df[(df[features] >= 0).all(axis=1)]
remainingobj = df.shape[0]

print("from ", objinthecatalog, "intial objects we have now", remainingobj)
print("object discarded:", objinthecatalog - remainingobj)

df_sampled = df.sample(frac=0.2, random_state=SEED).reset_index(drop=True)

train_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    df_sampled[features], df_sampled[target], train_size=train_size, random_state=SEED
)

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
