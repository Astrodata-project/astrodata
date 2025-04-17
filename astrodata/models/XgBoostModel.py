import xgboost as xgb

class XGBoostModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = xgb.XGBClassifier(*args, **kwargs)
    
    def build_model(self, *args, **kwargs):
        # Model is built in __init__
        pass

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, filepath):
        self.model.save_model(filepath)
    
    def load(self, filepath):
        self.model.load_model(filepath)
