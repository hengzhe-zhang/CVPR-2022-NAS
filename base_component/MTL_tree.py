import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor


class MultiTaskDecisionTree(DecisionTreeRegressor):
    """
    多任务学习决策树
    """

    def fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted="deprecated"):
        X, label = X[:, :10], X[:, -1]
        label = LabelEncoder().fit_transform(label)
        self.models = []
        step = len(X) // 8
        for i in range(8):
            # 目前问题：Random决策树无法提供随机性
            tree = DecisionTreeRegressor()
            # 正则化的问题
            tree.fit(X[label == i], y[label == i])
            self.models.append(tree)
        self.importance = np.mean([t.feature_importances_ for t in self.models], axis=0)
        return self

    @property
    def feature_importances_(self):
        return self.importance

    def predict(self, X, check_input=True):
        X, label = X[:, :10], X[:, -1]
        label = LabelEncoder().fit_transform(label)
        prediction = []
        step = len(X) // 8
        id = 0
        for i in range(8):
            tree = self.models[id]
            prediction.append(tree.predict(X[label == i]))
            id += 1
        return np.concatenate(prediction, axis=0)
