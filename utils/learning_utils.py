from multiprocessing import Pool
from typing import Callable

import numpy as np
import pandas as pd
import scipy
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from pathos.multiprocessing import ProcessingPool
from sklearn.model_selection import KFold, cross_val_predict


def kendalltau(y_predict, Y_test_k):
    return scipy.stats.stats.kendalltau(y_predict, Y_test_k)[0]


def score_evaluation(model, X_all_k, Y_all_k):
    """
    交叉验证评估工具类
    """
    scores = []
    for id in KFold(5).split(X_all_k):
        train_id, test_id = id
        ranker = model.fit(X_all_k[train_id], Y_all_k[train_id])
        scores.append(kendalltau(Y_all_k[test_id], ranker.predict(X_all_k[test_id])))
    return np.mean(np.array(scores))


def init_worker(function, data):
    function.data = data


def sklearn_tuner(
        model_class,
        space_config: [dict],
        X: np.ndarray,
        y: np.ndarray,
        metric: Callable,
        greater_is_better: bool = True,
        n_splits=5,
        max_iter=16,
        report=False
) -> (dict, pd.DataFrame):
    """
    基于HEBO的调参工具类
    """
    n_suggestions = 12
    pool = ProcessingPool(n_suggestions)
    space = DesignSpace().parse(space_config)
    opt = HEBO(space)
    for i in range(max_iter):
        rec = opt.suggest(n_suggestions=n_suggestions)
        sign = -1. if greater_is_better else 1.0
        scores = pool.map(simple_cross_validation, [(X, y, model_class, r[1].to_dict(), metric, n_splits)
                                                    for r in rec.iterrows()])
        # score_v = score_evaluation(model, X, y)
        opt.observe(rec, sign * np.array(scores))
        print('Iter %d, best metric: %g' % (i, sign * opt.y.min()))
    best_id = np.argmin(opt.y.reshape(-1))
    best_hyp = opt.X.iloc[best_id]
    df_report = opt.X.copy()
    df_report['metric'] = sign * opt.y
    if report:
        return best_hyp.to_dict(), df_report
    else:
        return best_hyp.to_dict()


def simple_cross_validation(parameters):
    """
    调参交叉验证工具类
    """
    # print('start single task!')
    X, y, model_class, parameter, metric, n_splits = parameters
    model = model_class(**parameter)
    pred = cross_val_predict(model, X, y, cv=KFold(n_splits=n_splits, shuffle=True), n_jobs=-1)
    score_v = np.nan_to_num(metric(y, pred))
    return score_v
