import tensorflow as tf
from keras.metrics import Metric

from astrodata.ml.metrics.BaseMetric import BaseMetric


class TensorflowMetric(Metric, BaseMetric):
    """
    Adapter class for Keras/TensorFlow metrics.

    Allows keras.metrics.Metric objects to be used as objects compatible
    with the BaseMetric interface while maintaining full Keras metric functionality.
    """

    def __init__(self, metric, name=None, greater_is_better=True, **kwargs):
        """
        Initialize a TensorflowMetric.

        Parameters
        ----------
        metric : keras.metrics.Metric
            A Keras metric instance (e.g., keras.metrics.Accuracy()).
        name : str, optional
            Optional name for the metric. Defaults to metric.name.
        greater_is_better : bool, optional
            If True, higher values are better. Defaults to True.
        **kwargs
            Additional keyword arguments passed to the parent Metric class.
        """
        if not isinstance(metric, Metric):
            raise TypeError("metric must be an instance of keras.metrics.Metric")

        # Call parent Metric.__init__ with appropriate parameters
        super().__init__(name=name or metric.name, **kwargs)

        self._metric = metric
        self._greater_is_better = greater_is_better

    def __call__(self, y_true, y_pred, **kwargs) -> float:
        """
        Compute the metric.

        Parameters
        ----------
        y_true : tensor-like
            Ground truth labels.
        y_pred : tensor-like
            Predicted labels or probabilities.
        **kwargs
            Additional keyword arguments passed to the metric function.

        Returns
        -------
        float
            The computed score.
        """
        self._metric.reset_state()
        return self._metric(y_true, y_pred, **kwargs).numpy()

    def get_name(self) -> str:
        """
        Return the name of the metric.

        Returns
        -------
        str
            The metric name.
        """
        return self.name

    @property
    def greater_is_better(self) -> bool:
        """
        Whether a higher metric value is better.

        Returns
        -------
        bool
            True if higher values are better; False otherwise.
        """
        return self._greater_is_better

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Update the metric state.

        Parameters
        ----------
        y_true : tensor-like
            Ground truth labels.
        y_pred : tensor-like
            Predicted labels or probabilities.
        sample_weight : tensor-like, optional
            Sample weights for the metric computation.

        Returns
        -------
        None
        """
        return self._metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    def result(self):
        """
        Compute the current metric result.

        Returns
        -------
        tensor
            The current metric value as a tensor.
        """
        return self._metric.result()

    def reset_state(self):
        """
        Reset the metric state.

        Returns
        -------
        None
        """
        return self._metric.reset_state()

    def __repr__(self):
        return f"TensorflowMetric(metric={self._metric}, name='{self._metric.name}', greater_is_better={self._greater_is_better})"

    def get_tf_metric(self):
        return self._metric
