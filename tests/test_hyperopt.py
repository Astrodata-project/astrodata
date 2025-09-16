import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from astrodata.ml.metrics.SklearnMetric import SklearnMetric
from astrodata.ml.model_selection.HyperOptSelector import HyperOptSelector
from astrodata.ml.models.SklearnModel import SklearnModel


def test_hyperopt_selector_val_split():
    pytest.importorskip("hyperopt", reason="hyperopt missing")
    from hyperopt import hp

    X, y = make_classification(
        n_samples=80, n_features=6, n_informative=4, n_redundant=0, random_state=0
    )

    space = {
        "model": SklearnModel(LogisticRegression, solver="liblinear", random_state=0),
        "C": hp.choice("C", [0.1, 1.0, 2.0]),
    }
    selector = HyperOptSelector(
        param_space=space,
        scorer=SklearnMetric(accuracy_score),
        use_cv=False,
        val_size=0.25,
        max_evals=5,
        random_state=0,
        metrics=[SklearnMetric(accuracy_score)],
    )

    selector.fit(X, y)
    best_model = selector.get_best_model()
    best_params = selector.get_best_params()
    best_metrics = selector.get_best_metrics()

    assert best_model is not None
    assert isinstance(best_params, dict) and "C" in best_params
    assert isinstance(best_metrics, dict) and "accuracy_score" in best_metrics
    yhat = best_model.predict(X)
    assert len(yhat) == len(y)
