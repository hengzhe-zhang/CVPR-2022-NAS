from scipy.stats import rankdata
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from dataset_loader import arch_list_train, train_list
from utils.learning_utils import kendalltau


class LRXGBRegressor(XGBRegressor):
    def fit(self, X, y, **kwargs):
        self.gp_model = Ridge()
        gp_prediction = self.gp_model.fit(X, y).predict(X)
        gp_prediction = rankdata(gp_prediction)
        super().fit(X, y - gp_prediction, **kwargs)
        return self

    def predict(self, X, **kwargs):
        gp_prediction = self.gp_model.predict(X)
        gp_prediction = rankdata(gp_prediction)
        xgb_prediction = super().predict(X)
        xgb_prediction = rankdata(xgb_prediction)
        return xgb_prediction + gp_prediction


if __name__ == '__main__':
    # regr = LRXGBRegressor(n_estimators=100, max_depth=1, objective='rank:pairwise', n_jobs=1,
    #                       learning_rate=0.8)
    # regr = RankSVM()
    # regr = LinearRegression()
    # regr = KNeighborsRegressor()
    # regr = XGBRegressor(n_estimators=2000, max_depth=1, objective='rank:pairwise', n_jobs=1,
    #                     learning_rate=0.8)
    regr = VotingRegressor([
        ('A', XGBRegressor(n_estimators=100, max_depth=1, objective='rank:pairwise', n_jobs=1,
                           learning_rate=0.8)),
        ('B', XGBRegressor(n_estimators=100, max_depth=1, objective='rank:pairwise', n_jobs=1,
                           learning_rate=0.8))
    ])
    # regr=DecisionTreeRegressor()
    # print('XGBRanker', np.mean(cross_val_score(
    #     regr,
    #     arch_list_train, train_list[0],
    #     scoring=make_scorer(kendalltau), n_jobs=1)))
    label = train_list[0]
    regr.fit(arch_list_train, label)
    print(kendalltau(label, regr.predict(arch_list_train)))
