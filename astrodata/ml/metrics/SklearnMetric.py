from astrodata.ml.metrics.BaseMetric import BaseMetric


class SklearnMetric(BaseMetric):
    """
    Adapter class for scikit-learn-style metrics.

    Allows metric functions (such as those from sklearn.metrics) to be used
    as objects compatible with the BaseMetric interface.
    """

    def __init__(self, metric, name=None, greater_is_better=True, **kwargs):
        """
        Initialize a SklearnMetric.

        Parameters
        ----------
        metric : callable
            A metric function (e.g., sklearn.metrics.accuracy_score).
        name : str, optional
            Optional name for the metric. Defaults to metric.__name__.
        greater_is_better : bool, optional
            If True, higher values are better. Defaults to True.
        **kwargs
            Additional default keyword arguments for the metric function.
        """
        self.metric = metric
        self.name = name if name is not None else metric.__name__
        self.default_kwargs = kwargs
        self.gib = greater_is_better

    def __call__(self, y_true, y_pred, **kwargs) -> float:
        """
        Compute the metric.

        Parameters
        ----------
        y_true : array-like
            Ground truth labels.
        y_pred : array-like
            Predicted labels or probabilities.
        **kwargs
            Additional keyword arguments passed to the metric function.

        Returns
        -------
        float
            The computed score.
        """
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.metric(y_true, y_pred, **all_kwargs)

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
        return self.gib
