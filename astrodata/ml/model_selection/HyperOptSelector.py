from ml.model_selection.BaseModelSelector import BaseModelSelector

# TODO


class HyperOptSelector(BaseModelSelector):
    def __init__(self):
        super().__init__()

    def fit(self, X, y, *args, **kwargs):
        pass

    def get_best_model(self):
        pass

    def get_best_params(self):
        pass

    def get_params(self, **kwargs) -> dict:
        pass
