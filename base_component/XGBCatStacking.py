import numpy as np
from catboost import CatBoostRanker
from scipy.stats import rankdata
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.datasets import make_regression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from base_component.ranksvm import RankSVM
from utils.learning_utils import kendalltau


class RankerNormalizer(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y, )
        return self

    def predict(self, X):
        label = self.model.predict(X)
        return rankdata(label)


class CatBoostPairwiseRanker(RegressorMixin, CatBoostRanker):

    def fit(self, X, y=None, group_id=None, **kwargs):
        group_id = np.ones_like(y).astype(int)
        self._loss_value_change = np.ones(X.shape[1])
        return super().fit(X, y, group_id, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)


class XCSkatingRanker(RegressorMixin, BaseEstimator):
    def __init__(self, weight=.0, hyper_parameters=None, post_ranking=True,
                 final_model='LR', full_stack=False, **parameters):
        self.full_stack = full_stack
        self.weight = weight
        if hyper_parameters != None:
            self.hyper_parameters = hyper_parameters
        else:
            self.hyper_parameters = parameters
        self.post_ranking = post_ranking
        self.final_model = final_model

    def fit(self, X, y=None, group_id=None, **kwargs):
        xgb_parameters = {}
        cat_parameters = {}
        elastic_net_parameters = {}
        for k, v in self.hyper_parameters.items():
            if 'XGB' in k:
                xgb_parameters[k.replace('XGB_', '')] = v
            if 'Cat' in k:
                cat_parameters[k.replace('Cat_', '')] = v
            if 'EN' in k:
                elastic_net_parameters[k.replace('EN_', '')] = v
        if self.post_ranking:
            xgb = RankerNormalizer(XGBRegressor(n_jobs=1, **xgb_parameters))
            cat = RankerNormalizer(CatBoostPairwiseRanker(verbose=False, thread_count=1, **cat_parameters))
        else:
            xgb = XGBRegressor(n_jobs=1, **xgb_parameters)
            cat = CatBoostPairwiseRanker(verbose=False, thread_count=1, **cat_parameters)

        if self.final_model == 'LR':
            final_model = LinearRegression()
        elif self.final_model == 'ElasticNet':
            final_model = ElasticNetCV(**elastic_net_parameters)
        elif self.final_model == 'SVM':
            final_model = RankSVM()
        else:
            raise Exception
        if self.full_stack:
            self.stacking_model = StackingRegressor(
                [('XGB', xgb), ('Cat', cat),
                 ('LR', LinearRegression()),
                 ], final_model, n_jobs=1
            )
        else:
            self.stacking_model = StackingRegressor(
                [('XGB', xgb), ('Cat', cat)], final_model, n_jobs=1
            )
        self.stacking_model.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        return self.stacking_model.predict(X)


if __name__ == '__main__':
    X, y = make_regression(random_state=0)
    r = XCSkatingRanker(weight=0.2, **{
        'XGB_objective': 'rank:pairwise',
        'Cat_loss_function': 'PairLogit',
        'full_stack': False,
        'final_model': 'LR',
    })
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    r.fit(x_train, y_train)
    print(kendalltau(y_test, r.predict(x_test)))
