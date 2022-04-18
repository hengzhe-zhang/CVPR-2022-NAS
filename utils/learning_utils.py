import gc
from typing import Callable

import numpy as np
import pandas as pd
import ray
import scipy
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO
from hebo.optimizers.hebo import HEBO
from pathos.multiprocessing import ProcessingPool
from pymoo.util.dominator import Dominator
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score


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


def get_pf_index(points: np.ndarray) -> np.ndarray:
    """
    获取PF上的数据点
    """
    dom_matrix = Dominator().calc_domination_matrix(points, None)
    is_optimal = (dom_matrix >= 0).all(axis=1)
    return is_optimal


def sklearn_tuner(
        model_class,
        space_config: [dict],
        X: np.ndarray,
        y: np.ndarray,
        metric: Callable,
        greater_is_better: bool = True,
        n_splits=5,
        max_iter=16,
        report=False,
        multi_objective=True
) -> (dict, pd.DataFrame):
    """
    基于HEBO的调参工具类
    """
    n_suggestions = 12
    ray.init(num_cpus=n_suggestions, _node_ip_address='127.0.0.1')
    pool = ProcessingPool(n_suggestions)
    space = DesignSpace().parse(space_config)
    if multi_objective:
        opt = GeneralBO(space, num_obj=5)
    else:
        opt = HEBO(space)
    X_id = ray.put(X)
    y_id = ray.put(y)
    for i in range(max_iter):
        rec = opt.suggest(n_suggestions=n_suggestions)
        sign = -1. if greater_is_better else 1.0
        # 贝叶斯优化
        # scores = pool.map(simple_cross_validation, [(X, y, model_class, r[1].to_dict(), metric, n_splits)
        #                                             for r in rec.iterrows()])
        scores = ray.get([simple_cross_validation.remote(X_id, y_id, model_class, r[1].to_dict(), metric, n_splits)
                          for r in rec.iterrows()])
        # score_v = score_evaluation(model, X, y)
        opt.observe(rec, sign * np.array(scores))
        print('Iter %d, best metric: %g' % (i, sign * opt.y.min()))
    pool.close()
    ray.shutdown()

    if multi_objective:
        parameters = opt.X[get_pf_index(opt.y)]
        return parameters.to_dict('records')
    else:
        best_id = np.argmin(opt.y.reshape(-1))
        best_hyp = opt.X.iloc[best_id]
        df_report = opt.X.copy()
        df_report['metric'] = sign * opt.y
        if report:
            return best_hyp.to_dict(), df_report
        else:
            return best_hyp.to_dict()


@ray.remote
def simple_cross_validation(X, y, model_class, parameter, metric, n_splits):
    """
    调参交叉验证工具类
    """
    # print('start single task!')
    # X, y, model_class, parameter, metric, n_splits = parameters
    all_score = []
    model = model_class(verbosity=0, **parameter)
    # pred = cross_val_predict(model, X, y, cv=KFold(n_splits=n_splits, shuffle=True), n_jobs=-1)
    # # Warning: 不同Batch的结果不可比
    # score_v = np.nan_to_num(metric(y, pred))
    score_v = cross_val_score(model, X, y, cv=KFold(shuffle=True), n_jobs=n_splits,
                              scoring=make_scorer(kendalltau))
    all_score.append(np.mean(score_v))
    gc.collect()
    # return np.mean(score_v)
    return score_v


if __name__ == '__main__':
    real = np.array([0, 12, 3, 18, 19, 16, 17, 10, 5, 7, 14, 8, 1, 9, 11, 4, 13, 2, 6, 15])
    predict = np.array([0, 12, 3, 18, 19, 16, 17, 10, 5, 7, 14, 8, 1, 9, 11, 4, 2, 13, 6, 15])
    real = np.concatenate([real, real + 30, real + 50])
    predict = np.concatenate([predict, predict + 30, predict + 50])
    print(kendalltau(real, predict))
    print(kendalltau(predict, real))
