"""
RankSVM 核心框架
"""
import itertools

import numpy as np
from evolutionary_forest.forest import spearman
from sklearn import svm
from sklearn.base import RegressorMixin
from sklearn.datasets import make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from base_component.gpnas import GPNASRegressor
from utils.learning_utils import kendalltau


def transform_pairwise(X: np.ndarray, y: np.ndarray):
    """
    构造Pairwise数据集
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(RegressorMixin, LogisticRegressionCV):
    def fit(self, X, y, **kwargs):
        """
        构造Pairwise数据集
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        尝试利用参数进行预测
        """
        if hasattr(self, 'coef_'):
            return np.dot(X, self.coef_.flatten())
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y, **kwargs):
        """
        评分函数
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return kendalltau(super(RankSVM, self).predict(X_trans), y_trans)


if __name__ == '__main__':
    X, y = make_regression(random_state=0)
    for rt in [0.2, 0.5, 0.8]:
        r = GPNASRegressor(rt)
        # r = LinearRegression()
        # X = StandardScaler().fit_transform(X)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        r.fit(x_train, y_train)
        print(kendalltau(y_test, r.predict(x_test)))
