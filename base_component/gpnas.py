import os
from paddleslim.nas import GPNAS
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split


class GPNASRegressor(RegressorMixin, BaseEstimator, GPNAS):
    def __init__(self, ratio=0.5, c_flag=2, m_flag=2, hp_mat=0.0000001, hp_cov=0.01):
        super().__init__(c_flag, m_flag)
        self.c_flag = c_flag
        self.m_flag = m_flag
        self.hp_mat = hp_mat
        self.hp_cov = hp_cov
        self.ratio = ratio

    def fit(self, X_train_k, Y_train_k):
        self.X_train_k = X_train_k
        self.Y_train_k = Y_train_k
        x_train, x_test, y_train, y_test = train_test_split(X_train_k, Y_train_k,
                                                            test_size=self.ratio, random_state=0)
        self.get_initial_mean(x_train, y_train)
        self.get_initial_cov(X_train_k)
        # 更新（训练）gpnas预测器超参数
        self.get_posterior_mean(x_test, y_test)
        return self

    def predict(self, X):
        return self.get_predict_jiont(X, self.X_train_k, self.Y_train_k)


class BaggingGP(GPNASRegressor):

    def fit(self, X_train_k, Y_train_k):
        self.bag = BaggingRegressor([GPNASRegressor() for _ in range(10)])
        self.bag.fit(X_train_k, Y_train_k)
        return self

    def predict(self, X):
        return self.bag.predict(X)
