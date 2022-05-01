import numpy as np
from catboost import CatBoostRanker
from deslib.des import KNORAE
from evolutionary_forest.forest import spearman
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.datasets import make_regression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from base_component.ranksvm import RankSVM


class CatBoostPairwiseRanker(RegressorMixin, CatBoostRanker):

    def fit(self, X, y=None, group_id=None, **kwargs):
        group_id = np.ones_like(y).astype(int)
        self._loss_value_change = np.ones(X.shape[1])
        return super().fit(X, y, group_id, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)


class XCSkatingRanker(RegressorMixin, BaseEstimator):
    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, X, y=None, group_id=None, **kwargs):
        xgb_parameters = {}
        cat_parameters = {}
        for k, v in self.parameters.items():
            if 'XGB' in k:
                xgb_parameters[k.replace('XGB_', '')] = v
            if 'Cat' in k:
                cat_parameters[k.replace('Cat_', '')] = v
        xgb = XGBRegressor(n_jobs=1, **xgb_parameters)
        cat = CatBoostPairwiseRanker(verbose=False, thread_count=1, **cat_parameters)
        self.stacking_model = StackingRegressor(
            [('XGB', xgb), ('Cat', cat)], KNORAE()
        )
        self.stacking_model.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        return self.stacking_model.predict(X)


if __name__ == '__main__':
    X, y = make_regression(random_state=0)
    r = XCSkatingRanker(**{
        'XGB_objective': 'rank:pairwise',
        'Cat_loss_function': 'PairLogit'
    })
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    r.fit(x_train, y_train)
    print(spearman(y_test, r.predict(x_test)))
