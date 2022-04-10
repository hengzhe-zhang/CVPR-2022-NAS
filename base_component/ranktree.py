import itertools

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR


class RankLearning():
    # form all pairwise combinations
    def pairwise_construction(self, X_train, y_train, classification=True):
        comb = itertools.combinations(range(X_train.shape[0]), 2)
        k = 0
        Xp, yp, diff = [], [], []
        for (i, j) in comb:
            if y_train[i] == y_train[j]:
                # skip if same target or different group
                continue
            Xp.append(X_train[i] - X_train[j])
            diff.append(y_train[i] - y_train[j])
            if classification:
                yp.append(np.sign(diff[-1]))
            else:
                yp.append(diff[-1])
            # output balanced classes
            if ((yp[-1] != (-1) ** k) and classification) \
                    or ((((-1) ** k) * yp[-1] > 0) and not classification):
                yp[-1] *= -1
                Xp[-1] *= -1
                diff[-1] *= -1
            k += 1
        return Xp, yp


class RankTree(BaseEstimator, RegressorMixin, RankLearning):
    """
    基于Rank的决策树算法
    """

    def fit(self, X, y, **kwargs):
        # self.model = LogisticRegression(solver='liblinear')
        self.model = LinearRegression()
        # self.model = SVR()
        X, y = self.pairwise_construction(X, y, isinstance(LogisticRegression, ClassifierMixin))
        X, y = np.array(X), np.array(y)
        self.avg_x = X.mean(axis=0).reshape(1, -1)
        self.model.fit(X, y)
        return self

    def predict(self, X, **kwargs):
        if isinstance(self.model, ClassifierMixin):
            return self.model.predict_proba(X - self.avg_x)[:, 1]
        else:
            return self.model.predict(X - self.avg_x)
