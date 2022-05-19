import numpy as np
from xgboost import XGBRanker

from base_component.gpnas import GPNASRegressor


class XGBRankerRegressor(XGBRanker):
    def fit(self, X, y, **kwargs):
        qid = np.ones(len(y))
        self.gp_model = GPNASRegressor()
        gp_prediction = self.gp_model.fit(X, y).predict(X)
        X = np.concatenate([X, np.reshape(gp_prediction, (-1, 1))], axis=1)
        return super().fit(X, y, qid=qid, **kwargs)

    def predict(self, X, **kwargs):
        gp_prediction = self.gp_model.predict(X)
        X = np.concatenate([X, np.reshape(gp_prediction, (-1, 1))], axis=1)
        return super().predict(X)
