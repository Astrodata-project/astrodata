"""
Tests for Keras/TensorFlow models.

Note: Keras is now integrated with TensorFlow (Keras 3+).
All TensorFlow/Keras model tests are located in test_tensorflow_model.py.

This file exists for backwards compatibility and to ensure test discovery
works for users searching for keras-specific tests.
"""

import pytest


def test_keras_model_exists():
    """Verify that TensorflowModel (Keras backend) can be imported."""
    pytest.importorskip("tensorflow", reason="tensorflow missing")
    pytest.importorskip("keras", reason="keras missing")

    from astrodata.ml.models.TensorflowModel import TensorflowModel

    assert TensorflowModel is not None


# Additional Keras-specific tests can be found in test_tensorflow_model.py
