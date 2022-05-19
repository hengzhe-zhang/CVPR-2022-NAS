# 假设对90%的数据进行训练，对10%的数据进行验证。
from itertools import product, chain

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from evolutionary_forest.forest import spearman
from paddle.io import Dataset
from paddle.nn import MSELoss, BCEWithLogitsLoss
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 自定义数据集
# 映射式(map-style)数据集需要继承paddle.io.Dataset
class SelfDefinedDataset(Dataset):
    def __init__(self, data_x, data_y, mode='train'):
        super(SelfDefinedDataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'predict':
            return self.data_x[idx]
        else:
            return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)


class RankLoss(MSELoss):
    def forward(self, y_pred, y_true):
        y_true = y_true.squeeze(-1)
        y_pred = y_pred.squeeze(-1)
        document_pairs_candidates = list(product(range(y_true.shape[0]), repeat=2))
        document_pairs_candidates = np.array(document_pairs_candidates)
        pairs_true = y_true[document_pairs_candidates]
        selected_pred = y_pred[document_pairs_candidates]

        # here we calculate the relative true relevance of every candidate pair
        true_diffs = pairs_true[:, 0] - pairs_true[:, 1]
        pred_diffs = selected_pred[:, 0] - selected_pred[:, 1]

        the_mask = (true_diffs > 0) & (~paddle.isinf(true_diffs))
        pred_diffs = pred_diffs[the_mask]

        true_diffs = (true_diffs > 0).cast('float32')
        true_diffs = true_diffs[the_mask]

        return BCEWithLogitsLoss()(pred_diffs, true_diffs)


class RankNetPaddle(BaseEstimator, RegressorMixin):
    def __init__(self, number_of_hidden_layer=0, hidden_layer_size=16, dropout_rate=.0,
                 learning_rate=0.1, l2_regularization=1e-5, batch_size=16,
                 epochs=100, verbose=False):
        self.batch_size = round(batch_size)
        self.l2_regularization = l2_regularization
        self.learning_rate = learning_rate
        self.number_of_hidden_layer = round(number_of_hidden_layer)
        self.hidden_layer_size = round(hidden_layer_size)
        self.dropout_rate = dropout_rate
        self.epochs = round(epochs)
        self.verbose = verbose

    def fit(self, X, y, **kwargs):
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        traindataset = SelfDefinedDataset(X, y)
        train_loader = paddle.io.DataLoader(traindataset, batch_size=128, shuffle=True)

        model = paddle.nn.Sequential(
            paddle.nn.Linear(X.shape[1], self.hidden_layer_size),
            paddle.nn.Dropout(self.dropout_rate),
            *list(chain.from_iterable([[
                paddle.nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
                paddle.nn.Dropout(self.dropout_rate)
            ]
                for _ in range(self.number_of_hidden_layer)])),
            paddle.nn.Linear(self.hidden_layer_size, 1)
        )

        model = paddle.Model(model)

        optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                          learning_rate=self.learning_rate,
                                          weight_decay=self.l2_regularization)
        loss = RankLoss()

        model.prepare(optimizer, loss)
        model.fit(train_loader, epochs=self.epochs, verbose=self.verbose,
                  batch_size=self.batch_size)
        self.model = model
        return self

    def predict(self, X, y=None):
        X = X.astype(np.float32)
        traindataset = SelfDefinedDataset(X, None, mode='predict')
        return np.array(self.model.predict(traindataset)).flatten()


def training_task():
    x, y = load_boston(return_X_y=True)

    r = RankNetPaddle(number_of_hidden_layer=1, hidden_layer_size=16,
                      dropout_rate=0.2, learning_rate=0.1, l2_regularization=1e-5, batch_size=32)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    r.fit(x_train, y_train)
    print(spearman(y_test, r.predict(x_test)))


if __name__ == '__main__':
    training_task()
