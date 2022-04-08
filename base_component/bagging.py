import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


class CustomBaggingRegressor(RegressorMixin, BaseEstimator):
    """
    一个自定义的Bagging算法，默认Bagging算法LTR会出错
    """

    def __init__(self, model, size=10):
        self.model = model
        self.size = size

    def fit(self, X_train, y_train, seed=None):
        self.X_train = X_train
        self.N, self.D = X_train.shape
        self.y_train = y_train
        self.seed = seed
        self.trees = []

        for _ in range(self.size):
            sample = np.random.choice(np.arange(self.N), size=self.N, replace=True)
            X_train_b = self.X_train[sample]
            y_train_b = self.y_train[sample]
            tree: BaseEstimator = self.model
            tree.fit(X_train_b, y_train_b)
            self.trees.append(tree)
        return self

    def predict(self, X_test):
        y_test_hats = np.empty((len(self.trees), len(X_test)))
        for i, tree in enumerate(self.trees):
            y_test_hats[i] = tree.predict(X_test)
        return np.mean(y_test_hats, axis=0)
