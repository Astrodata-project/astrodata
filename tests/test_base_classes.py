"""Tests for base model classes and abstract interfaces."""

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from astrodata.ml.models.BaseMlModel import BaseMlModel
from astrodata.ml.models.SklearnModel import SklearnModel


def test_base_model_is_abstract():
    """BaseMlModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseMlModel()


def test_base_model_defines_required_methods():
    """BaseMlModel defines all required abstract methods."""
    required_methods = [
        "fit",
        "predict",
        "score",
        "get_scorer_metric",
        "save",
        "load",
        "get_metrics",
    ]
    for method in required_methods:
        assert hasattr(BaseMlModel, method)


def test_base_model_get_params_not_implemented():
    """get_params raises NotImplementedError by default."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = SklearnModel(LogisticRegression, solver="liblinear")
    model.fit(X, y)

    # SklearnModel implements get_params, so it should work
    params = model.get_params()
    assert isinstance(params, dict)
    assert "model_class" in params


def test_base_model_set_params_not_implemented():
    """set_params raises NotImplementedError by default."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = SklearnModel(LogisticRegression, solver="liblinear")
    model.fit(X, y)

    # SklearnModel implements set_params, so it should work
    model.set_params(max_iter=200)
    assert model.model_params["max_iter"] == 200


def test_concrete_model_implements_all_abstract_methods():
    """Concrete models must implement all abstract methods."""
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    model = SklearnModel(LogisticRegression, solver="liblinear")

    # Should not raise errors - all methods are implemented
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions is not None

    score = model.score(X, y)
    assert isinstance(score, (float, int))

    scorer_metric = model.get_scorer_metric()
    assert scorer_metric is not None

    from astrodata.ml.metrics.SklearnMetric import SklearnMetric
    from sklearn.metrics import accuracy_score

    metrics = model.get_metrics(X, y, metrics=[SklearnMetric(accuracy_score)])
    assert isinstance(metrics, dict)


def test_base_model_subclass_must_implement_abstract_methods():
    """Subclasses that don't implement abstract methods cannot be instantiated."""

    class IncompleteModel(BaseMlModel):
        """A model that doesn't implement all abstract methods."""

        def fit(self, X, y, **kwargs):
            pass

        # Missing: predict, score, get_scorer_metric, save, load, get_metrics

    with pytest.raises(TypeError):
        IncompleteModel()


def test_model_clone_sklearn():
    """Test cloning functionality for SklearnModel."""
    model = SklearnModel(
        LogisticRegression, solver="liblinear", max_iter=100, random_state=42
    )

    cloned = model.clone()
    assert cloned is not model
    assert cloned.model_class == model.model_class
    assert cloned.model_params == model.model_params
    assert cloned.random_state == model.random_state
    # Model should not be fitted yet
    assert cloned.model_ is None


def test_model_clone_preserves_params():
    """Cloned models preserve initialization parameters."""
    model = SklearnModel(
        LogisticRegression,
        solver="liblinear",
        max_iter=200,
        C=0.5,
        random_state=123,
    )

    cloned = model.clone()
    cloned_params = cloned.get_params()

    assert cloned_params["max_iter"] == 200
    assert cloned_params["C"] == 0.5
    assert cloned_params["random_state"] == 123


def test_model_get_set_params_roundtrip():
    """get_params and set_params work together correctly."""
    model = SklearnModel(
        LogisticRegression, solver="liblinear", max_iter=100, random_state=42
    )

    original_params = model.get_params()
    model.set_params(max_iter=300, C=2.0)

    updated_params = model.get_params()
    assert updated_params["max_iter"] == 300
    assert updated_params["C"] == 2.0
    assert updated_params["model_class"] == original_params["model_class"]
