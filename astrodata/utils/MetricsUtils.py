from sklearn import metrics as sklearn_metrics


def get_loss_func(model):
    # XGBoost models: use .get_xgb_params()["objective"]
    if "xgboost" in type(model).__module__:
        obj = model.get_xgb_params().get("objective", None)
        # Map XGBoost objectives to sklearn loss functions
        mapping = {
            "binary:logistic": "log_loss",
            "multi:softprob": "log_loss",  # For probabilities
            "multi:softmax": "accuracy_score",  # XGB softmax returns classes
            "reg:squarederror": "mean_squared_error",
            "reg:squaredlogerror": "mean_squared_log_error",
            "reg:logistic": "log_loss",
            "reg:absoluteerror": "mean_absolute_error",
        }
        sklearn_loss_name = mapping.get(obj, None)
        if sklearn_loss_name and hasattr(sklearn_metrics, sklearn_loss_name):
            return getattr(sklearn_metrics, sklearn_loss_name)

    # scikit-learn models: try 'loss' param
    params = model.get_params()
    loss = params.get("loss", None)
    # Map sklearn loss param values to metric function names
    loss_mapping = {
        "squared_error": "mean_squared_error",
        "squared_loss": "mean_squared_error",  # old name
        "log_loss": "log_loss",
        "hinge": "hinge_loss",
        "epsilon_insensitive": "mean_absolute_error",
        "absolute_error": "mean_absolute_error",
        "huber": "mean_squared_error",  # no direct sklearn metric for huber
        "poisson": "mean_poisson_deviance",
    }
    loss_func_name = loss_mapping.get(loss, loss)
    if loss_func_name and hasattr(sklearn_metrics, loss_func_name):
        return getattr(sklearn_metrics, loss_func_name)

    # Fallback: None
    return None
