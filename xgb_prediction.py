import os

import dill
import numpy as np
from catboost import CatBoostRanker
from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.utils import get_feature_importance, feature_append
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from base_component.bagging import CustomBaggingRegressor
from base_component.ranksvm import RankSVM
from base_component.ranktree import RankTree
from dataset_loader import *
# 加载训练集
from hyperparameters import history_best_parameters
from utils.notify_utils import notify
from utils.learning_utils import sklearn_tuner, kendalltau

os.environ['OMP_NUM_THREADS'] = "4"
os.environ['MKL_NUM_THREADS'] = "4"
os.environ['NUMEXPR_NUM_THREADS'] = "4"
use_xgboost = True

xgb_search_space = [
    {'name': 'booster', 'type': 'cat', 'categories': ['gbtree', 'dart', 'gblinear']},
    {'name': 'n_estimators', 'type': 'int', 'lb': 10, 'ub': 1000},
    # {'name': 'eval_metric', 'type': 'cat', 'categories': ['logloss']},
    # L2 正则化
    {'name': 'reg_lambda', 'type': 'pow', 'lb': 1e-6, 'ub': 1e3},
    # L1正则化
    {'name': 'alpha', 'type': 'pow', 'lb': 1e-6, 'ub': 1e2},
    # 允许分裂的最小损失
    {'name': 'gamma', 'type': 'num', 'lb': 0, 'ub': 0.5},
    # 允许分裂的最小权重
    {'name': 'min_child_weight', 'type': 'num', 'lb': 0, 'ub': 10},
    # 学习率
    {'name': 'eta', 'type': 'pow', 'lb': 1e-8, 'ub': 1},
    # 决策树深度
    {'name': 'max_depth', 'type': 'int', 'lb': 1, 'ub': 10},
    {'name': 'n_jobs', 'type': 'cat', 'categories': [1]},
    {'name': 'objective', 'type': 'cat', 'categories': ['rank:pairwise', 'reg:squarederror']},
    # 降采样
    {'name': 'colsample_bytree', 'type': 'num', 'lb': 0.3, 'ub': 1},
    {'name': 'colsample_bylevel', 'type': 'num', 'lb': 0.3, 'ub': 1},
    {'name': 'subsample', 'type': 'num', 'lb': 0.3, 'ub': 1},
]

catboost_params = [
    {'name': 'depth', 'type': 'int', 'lb': 1, 'ub': 10},
    {'name': 'iterations', 'type': 'int', 'lb': 10, 'ub': 1000},
    {'name': 'learning_rate', 'type': 'pow', 'lb': 1e-3, 'ub': 1},
    {'name': 'l2_leaf_reg', 'type': 'pow', 'lb': 1e-6, 'ub': 1e3},
    {'name': 'loss_function', 'type': 'cat', 'categories': ['PairLogit', 'PairLogitPairwise',
                                                            'YetiRank']},
    {'name': 'thread_count', 'type': 'cat', 'categories': [1]},
    {'name': 'verbose', 'type': 'cat', 'categories': [False]},
]

"""
超参数搜索结果
基准平均分 0.77589898989899
超参数集成后平均分：0.7775959595959597
"""


class CatBoostPairwiseRanker(CatBoostRanker):

    def fit(self, X, y=None, group_id=None, **kwargs):
        group_id = np.ones_like(y)
        return super().fit(X, y, group_id, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)


@notify
def simple_cv_task(fe=False, custom_fe=False, final_prediction=False):
    """
    交叉验证测试
    """
    # 特征构造之后平均分 0.7511313131313132
    # 平均分 0.7517373737373738
    ef = EvolutionaryForestRegressor(max_height=5, normalize=True, select='AutomaticLexicase',
                                     gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                     base_learner='Random-DT', verbose=True, n_process=1)
    # 这个任务主要用来计算平均分数
    scores = []
    prediction_results = []
    for i in range(len(train_list[:])):
        X_all_k, Y_all_k = np.array(arch_list_train), np.array(train_list[i])

        if custom_fe:
            data = [X_all_k, PCA(n_components=5).fit_transform(X_all_k)]
            X_all_k = np.concatenate(data, axis=1)
        if fe:
            ef.lazy_init(X_all_k)
            with open(f'model_{i}.pkl', 'rb') as f:
                ef = dill.load(f)
            code_importance_dict = get_feature_importance(ef, simple_version=False)
            print('number of features', len(code_importance_dict))
            X_all_k = feature_append(ef, X_all_k,
                                     list(code_importance_dict.keys())[:2],
                                     only_new_features=False)
        # BaggingRegressor(XGBRegressor(**history_best_parameters[i]), n_jobs=-1).fit(X_all_k, Y_all_k).predict(X_all_k)
        score = np.mean(cross_val_score(XGBRegressor(),
                                        X_all_k, Y_all_k,
                                        scoring=make_scorer(kendalltau), n_jobs=-1))
        # score = np.mean(cross_val_score(XGBRegressor(**history_best_parameters[i]),
        #                                 X_all_k, Y_all_k,
        #                                 scoring=make_scorer(kendalltau), n_jobs=-1))
        print('XGBRanker', score)
        scores.append(score)

        if final_prediction:
            # 最终预测
            xgb = XGBRegressor(**history_best_parameters[i])
            test_data_np = np.array(test_arch_list)
            xgb.fit(X_all_k, Y_all_k)
            if fe:
                test_data_np = feature_append(ef, test_data_np,
                                              list(code_importance_dict.keys())[:2],
                                              only_new_features=False)
            prediction = xgb.predict(test_data_np)
            prediction_results.append(prediction)
    print('平均分', np.mean(scores))
    keys = list(test_data.keys())
    data_print(keys, prediction_results)


@notify
def tuning_task(tuning=True):
    """
    调参+输出结果
    """
    keys = list(test_data.keys())
    prediction_results = []
    scores = []
    for i in range(len(train_list[:])):
        X_all_k, Y_all_k = np.array(arch_list_train), np.array(train_list[i])
        core_class = XGBRegressor if use_xgboost else CatBoostPairwiseRanker
        if tuning:
            """
            目前来看，调参效果最好，收益最大
            """
            search_space = xgb_search_space if use_xgboost else catboost_params
            best_parameter = sklearn_tuner(core_class, search_space, X_all_k, Y_all_k, metric=kendalltau,
                                           max_iter=16)
            print(best_parameter)
            print('XGBRanker', np.mean(cross_val_score(core_class(**best_parameter),
                                                       X_all_k, Y_all_k,
                                                       scoring=make_scorer(kendalltau), n_jobs=-1)))
            xgb = core_class(**best_parameter).fit(X_all_k, Y_all_k)
        else:
            """
            可以考虑的思路：
            1. Ensemble 目前无提升效果 线上分数：0.77526	
            2. RankSVM 离线效果不行
            3. 尝试一下RankDT
            """
            mean_score = np.mean(cross_val_score(
                core_class(**history_best_parameters['A'][i])
                # XGBRegressor(objective='rank:pairwise', n_jobs=1)
                # RankTree()
                , X_all_k, Y_all_k,
                scoring=make_scorer(kendalltau)))
            scores.append(mean_score)
            print('XGBRanker', mean_score)
            xgb = core_class(**history_best_parameters['A'][i]).fit(X_all_k, Y_all_k)
        prediction = xgb.predict(np.array(test_arch_list))
        prediction_results.append(prediction)
    print(np.mean(np.array(scores)))

    data_print(keys, prediction_results)


def data_print(keys, prediction_results):
    # 排序并输出结果
    for i, data in enumerate(rankdata(prediction_results, axis=1) - 1):
        for id, x in enumerate(data):
            test_data[keys[id]][name_list[i]] = int(x)
    with open('./CVPR_2022_NAS_Track2_submit_A.json', 'w') as f:
        json.dump(test_data, f)
    print('Ready to save results!')


"""
基于关系对的Baseline XGBoost
任务 0.1953939393939394
任务 0.7175757575757578
任务 0.720888888888889
任务 0.7418181818181819
任务 0.6893737373737374
任务 0.541818181818182
任务 0.7282424242424244
任务 0.7185454545454546
总分 0.6317070707070708

基于关系对的Baseline Logistic Regression
任务 0.2418585858585859
任务 0.7166060606060608
任务 0.7134545454545456
任务 0.7786155104498116
任务 0.623919191919192
任务 0.41389898989899
任务 0.693818181818182
任务 0.7541818181818182
总分 0.6170441105233982

调参后的结果
任务 0.2656969696969697
任务 0.861818181818182
任务 0.8880808080808082
任务 0.9516767676767678
任务 0.881939393939394
任务 0.6650505050505051
任务 0.9115959595959598
任务 0.7949090909090909
总分 0.7775959595959597
"""

if __name__ == '__main__':
    tuning_task(tuning=False)
    # simple_cv_task(custom_fe=False)
