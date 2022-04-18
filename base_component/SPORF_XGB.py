import numpy as np
from xgboost import XGBRegressor


def data_augmentation(X):
    a_index = np.arange(1, X.shape[1], 2)
    np.random.shuffle(a_index)
    X[np.arange(1, X.shape[1], 2)] = X[a_index]
    X[np.arange(2, X.shape[1], 2)] = X[a_index + 1]
    return X

"""
目前来看，随机扰动会导致模型效果变差
0.7513446893787574

目前还有哪些思路：
1. Ensemble Learning 增加随机性
2. 增加更多的Ensemble Learning Option
"""
class SPORFXGB(XGBRegressor):
    def fit(self, X, y, **kwargs):
        super().fit(np.concatenate([X, data_augmentation(X)], axis=0),
                    np.concatenate([y for _ in range(2)], axis=0), **kwargs)
        return self
