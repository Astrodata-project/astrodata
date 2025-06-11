from astrodata.ml.metrics.BaseMetric import BaseMetric


class SklearnMetric(BaseMetric):
    def __init__(self, metric, name=None, greater_is_better=True, **kwargs):
        """
        metric: a callable (e.g. sklearn.metrics.accuracy_score)
        name: optional name for the metric, defaults to metric.__name__
        kwargs: default keyword arguments for the metric function
        """
        self.metric = metric
        self.name = name if name is not None else metric.__name__
        self.default_kwargs = kwargs
        self.gib = greater_is_better

    def __call__(self, y_true, y_pred, **kwargs):
        """
        Call the metric function with y_true, y_pred, and any kwargs.
        """
        all_kwargs = {**self.default_kwargs, **kwargs}
        return self.metric(y_true, y_pred, **all_kwargs)

    def get_name(self):
        return self.name

    @property
    def greater_is_better(self):
        return self.gib
