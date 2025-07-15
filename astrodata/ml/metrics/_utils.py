from typing import Callable, Optional

from sklearn import metrics as sklearn_metrics


def get_loss_func(
    model, xgb_mapping: dict = None, skl_mapping: dict = None
) -> Optional[Callable]:
    """
    Given a model, try to return the appropriate sklearn.metrics loss function.

    Args:
        model: Fitted model instance (e.g., XGBoost or scikit-learn estimator).
        xgb_mapping: Optionally supply or extend the default XGBoost mapping.
        skl_mapping: Optionally supply or extend the default sklearn mapping.

    Returns:
        Callable (metric function) or None.
    """
    # --- XGBoost models ---
    if "xgboost" in type(model).__module__:
        try:
            obj = model.get_xgb_params().get("objective", None)
        except AttributeError:
            obj = None
        mapping = {
            "binary:logistic": "log_loss",
            "multi:softprob": "log_loss",  # For probabilities
            "multi:softmax": "accuracy_score",  # Softmax returns classes
            "reg:squarederror": "mean_squared_error",
            "reg:squaredlogerror": "mean_squared_log_error",
            "reg:logistic": "log_loss",
            "reg:absoluteerror": "mean_absolute_error",
        }
        if xgb_mapping is not None:
            mapping.update(xgb_mapping)
        sklearn_loss_name = mapping.get(obj, None)
        if sklearn_loss_name and hasattr(sklearn_metrics, sklearn_loss_name):
            return getattr(sklearn_metrics, sklearn_loss_name)

    # --- scikit-learn models ---
    params = {}
    if hasattr(model, "get_params"):
        try:
            params = model.get_params()
        except Exception:
            params = {}
    loss = params.get("loss", None)
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
    if skl_mapping is not None:
        loss_mapping.update(skl_mapping)
    loss_func_name = loss_mapping.get(loss, loss)
    if loss_func_name and hasattr(sklearn_metrics, loss_func_name):
        return getattr(sklearn_metrics, loss_func_name)

    # Fallback
    return None
