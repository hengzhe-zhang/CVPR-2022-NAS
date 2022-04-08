import os

import dill
from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.utils import get_feature_importance, feature_append
from scipy.stats import rankdata
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from base_component.ranksvm import RankSVM
from dataset_loader import *
# 加载训练集
from utils.notify_utils import notify
from utils import sklearn_tuner, kendalltau

os.environ['OMP_NUM_THREADS'] = "4"
os.environ['MKL_NUM_THREADS'] = "4"
os.environ['NUMEXPR_NUM_THREADS'] = "4"

xgb_search_space = [
    {'name': 'booster', 'type': 'cat', 'categories': ['gbtree', 'dart']},
    {'name': 'n_estimators', 'type': 'int', 'lb': 10, 'ub': 1000},
    # {'name': 'eval_metric', 'type': 'cat', 'categories': ['logloss']},
    {'name': 'reg_lambda', 'type': 'pow', 'lb': 1e-5, 'ub': 1e3},
    {'name': 'alpha', 'type': 'pow', 'lb': 1e-5, 'ub': 1e2},
    {'name': 'gamma', 'type': 'num', 'lb': 0, 'ub': 0.5},
    {'name': 'eta', 'type': 'pow', 'lb': 1e-8, 'ub': 1},
    {'name': 'max_depth', 'type': 'int', 'lb': 1, 'ub': 10},
    {'name': 'n_jobs', 'type': 'cat', 'categories': [1]},
    {'name': 'objective', 'type': 'cat', 'categories': ['rank:pairwise', 'reg:squarederror', 'reg:squaredlogerror',
                                                        'reg:pseudohubererror']},
]

"""
超参数搜索结果
平均分 0.77589898989899
"""
history_best_parameters = [
    {'n_estimators': 836, 'reg_lambda': 1.1435108409810695, 'alpha': 0.0003810556846692025,
     'gamma': 0.48345394186065116, 'eta': 0.019143432191868336, 'max_depth': 2, 'booster': 'dart', 'n_jobs': 1,
     'objective': 'rank:pairwise'},
    {'n_estimators': 669, 'reg_lambda': 134.4703888122001, 'alpha': 1.5803441968957594e-05,
     'gamma': 0.10259182374809228, 'eta': 0.6240626868110374, 'max_depth': 1, 'booster': 'gbtree', 'n_jobs': 1,
     'objective': 'rank:pairwise'},
    {'n_estimators': 599, 'reg_lambda': 0.0005235820380736115, 'alpha': 4.99856695873406, 'gamma': 0.4994256973611675,
     'eta': 0.9986555114477149, 'max_depth': 1, 'booster': 'gbtree', 'n_jobs': 1, 'objective': 'rank:pairwise'},
    {'n_estimators': 964, 'reg_lambda': 7.696360042725231, 'alpha': 4.4057606872680095, 'gamma': 0.0680825437611732,
     'eta': 0.48218997759952864, 'max_depth': 1, 'booster': 'gbtree', 'n_jobs': 1, 'objective': 'reg:squarederror'},
    {'n_estimators': 772, 'reg_lambda': 21.14820239721931, 'alpha': 4.0106315847522485, 'gamma': 0.042576935885660444,
     'eta': 0.7141108893391953, 'max_depth': 1, 'booster': 'gbtree', 'n_jobs': 1, 'objective': 'rank:pairwise'},
    {'n_estimators': 868, 'reg_lambda': 69.54127652200178, 'alpha': 6.487164551887698, 'gamma': 0.13458772975261038,
     'eta': 0.6637663632399583, 'max_depth': 1, 'booster': 'gbtree', 'n_jobs': 1, 'objective': 'rank:pairwise'},
    {'n_estimators': 965, 'reg_lambda': 84.86209328863855, 'alpha': 0.0006730206682587948, 'gamma': 0.4255841168344793,
     'eta': 0.6075092116164166, 'max_depth': 1, 'booster': 'gbtree', 'n_jobs': 1, 'objective': 'rank:pairwise'},
    {'n_estimators': 550, 'reg_lambda': 0.026646704851822713, 'alpha': 0.2501184370940979, 'gamma': 0.21204172083708014,
     'eta': 0.13671862865207857, 'max_depth': 2, 'booster': 'dart', 'n_jobs': 1, 'objective': 'rank:pairwise'},
]


@notify
def simple_cv_task(fe=False, custom_fe=False):
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
    for i in range(len(train_list[:])):
        X_all_k, Y_all_k = np.array(arch_list_train), np.array(train_list[i])

        if custom_fe:
            data = [X_all_k]
            X_all_k = np.concatenate(data, axis=1)
        if fe:
            ef.lazy_init(X_all_k)
            with open(f'model_{i}.pkl', 'rb') as f:
                ef = dill.load(f)
            code_importance_dict = get_feature_importance(ef, simple_version=False)
            print('number of features', len(code_importance_dict))
            X_all_k = feature_append(ef, X_all_k,
                                     list(code_importance_dict.keys())[:len(code_importance_dict) // 10],
                                     only_new_features=False)
        # BaggingRegressor(XGBRegressor(**history_best_parameters[i]), n_jobs=-1).fit(X_all_k, Y_all_k).predict(X_all_k)
        score = np.mean(cross_val_score(XGBRegressor(**history_best_parameters[i]),
                                        X_all_k, Y_all_k,
                                        scoring=make_scorer(kendalltau), n_jobs=-1))
        print('XGBRanker', score)
        scores.append(score)
    print('平均分', np.mean(scores))


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
        if tuning:
            best_parameter = sklearn_tuner(XGBRegressor, xgb_search_space, X_all_k, Y_all_k, metric=kendalltau)
            print(best_parameter)
            print('XGBRanker', np.mean(cross_val_score(XGBRegressor(**best_parameter),
                                                       X_all_k, Y_all_k,
                                                       scoring=make_scorer(kendalltau), n_jobs=-1)))
            xgb = XGBRegressor(**best_parameter).fit(X_all_k, Y_all_k)
        else:
            """
            可以考虑的思路：
            1. Ensemble 目前无提升效果 线上分数：0.77526	
            2. RankSV 离线效果不行
            """
            mean_score = np.mean(cross_val_score(
                # CustomBaggingRegressor(XGBRegressor(**history_best_parameters[i]))
                # XGBRegressor(objective='rank:pairwise', n_jobs=1)
                RankSVM()
                , X_all_k, Y_all_k,
                scoring=make_scorer(kendalltau), n_jobs=-1))
            scores.append(mean_score)
            print('XGBRanker', mean_score)
            # xgb = CustomBaggingRegressor(XGBRegressor(**history_best_parameters[i])).fit(X_all_k, Y_all_k)
            xgb = RankSVM().fit(X_all_k, Y_all_k)
        prediction = xgb.predict(np.array(test_arch_list))
        prediction_results.append(prediction)
    print(np.mean(np.array(scores)))

    # 排序并输出结果
    for i, data in enumerate(rankdata(prediction_results, axis=1) - 1):
        for id, x in enumerate(data):
            test_data[keys[id]][name_list[i]] = int(x)

    with open('./CVPR_2022_NAS_Track2_submit_A.json', 'w') as f:
        json.dump(test_data, f)
    print('Ready to save results!')


if __name__ == '__main__':
    tuning_task(tuning=False)
    # simple_cv_task(fe=True)
