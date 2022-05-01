import numpy as np
from evolutionary_forest.forest import spearman
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree, KNeighborsRegressor


class KNNLTR(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # Rank 0最佳
        ids = np.argsort(-1 * y)
        self.tree = KDTree(X[ids])
        self.y = y[ids]
        return self

    def predict(self, X):
        dis, ind = self.tree.query(X)
        ids = np.lexsort((dis.flatten(), ind.flatten()))
        ids = np.argsort(ids)
        # ids = ind.flatten()
        return -1 * ids


if __name__ == '__main__':
    # X = np.random.rand(100, 2)
    # Y = np.sum(X * 5, axis=1)
    X, Y = load_boston(return_X_y=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    knn = KNNLTR()
    # for i in range(1, 10):
    #     knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, Y_train)
    Y_preds = knn.predict(X_test)
    print(spearman(Y_preds, Y_test))
