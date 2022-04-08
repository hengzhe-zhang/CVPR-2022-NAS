import itertools

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RankTree(DecisionTreeClassifier):
    """
    基于Rank的决策树算法
    """

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"):
        X, y = self.pairwise_construction(X, y)
        super().fit(X, y, sample_weight, check_input, X_idx_sorted)
        return self

    # form all pairwise combinations
    def pairwise_construction(self, X_train, y_train):
        comb = itertools.combinations(range(X_train.shape[0]), 2)
        k = 0
        Xp, yp, diff = [], [], []
        for (i, j) in comb:
            if y_train[i] == y_train[j]:
                # skip if same target or different group
                continue
            Xp.append(X_train[i] - X_train[j])
            diff.append(y_train[i] - y_train[j])
            yp.append(np.sign(diff[-1]))
            # output balanced classes
            if yp[-1] != (-1) ** k:
                yp[-1] *= -1
                Xp[-1] *= -1
                diff[-1] *= -1
            k += 1
        return Xp, yp
