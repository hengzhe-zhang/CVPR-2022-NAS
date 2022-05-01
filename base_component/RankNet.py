from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from evolutionary_forest.forest import spearman
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss, BCEWithLogitsLoss


class ClassifierModule(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim=16,
            output_dim=1,
            dropout=0.75,
    ):
        super(ClassifierModule, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.dropout(X)
        X = F.relu(self.hidden2(X))
        X = self.dropout(X)
        X = self.output(X).squeeze(-1)
        return X


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RankNetLoss(_Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_true = y_true.squeeze(-1)
        document_pairs_candidates = list(product(range(y_true.shape[0]), repeat=2))

        pairs_true = y_true[document_pairs_candidates]
        selected_pred = y_pred[document_pairs_candidates]

        # here we calculate the relative true relevance of every candidate pair
        true_diffs = pairs_true[:, 0] - pairs_true[:, 1]
        pred_diffs = selected_pred[:, 0] - selected_pred[:, 1]

        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))
        pred_diffs = pred_diffs[the_mask]

        true_diffs = (true_diffs > 0).type(torch.float32)
        true_diffs = true_diffs[the_mask]

        return BCEWithLogitsLoss()(pred_diffs, true_diffs)


class RankNetRanker(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        X = X.astype(np.float32)
        y = y.astype(np.float32).reshape(-1, 1)
        self.std = StandardScaler()
        self.std.fit(X)
        X = self.std.transform(X)
        y = StandardScaler().fit_transform(y)
        net = NeuralNetRegressor(
            ClassifierModule,
            max_epochs=200,
            criterion=RankNetLoss,
            lr=0.01,
            device=device,
            optimizer=torch.optim.Adam,
            callbacks=[EarlyStopping(patience=50)],
            module__input_dim=X.shape[1],
            verbose=True
        )
        net.fit(X, y)
        self.net = net
        return self

    def predict(self, X):
        X = X.astype(np.float32)
        X = self.std.transform(X)
        return self.net.predict(X)


if __name__ == '__main__':
    # X, y = make_classification()
    X, y = make_regression(random_state=0)
    r = RankNetRanker()
    # r = XGBRegressor('rank:pairwise', n_jobs=1)
    # r = XGBRegressor()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    r.fit(x_train, y_train)
    print(spearman(y_test, r.predict(x_test)))
