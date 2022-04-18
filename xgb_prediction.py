import copy
import os

import dill
from catboost import CatBoostRanker
from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.utils import get_feature_importance, feature_append
from lightgbm import LGBMRanker
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from xgboost import XGBRegressor, XGBRanker

from dataset_loader import *
# 加载训练集
from hyperparameters import history_best_parameters
from utils.learning_utils import sklearn_tuner, kendalltau
from utils.notify_utils import notify

os.environ['OMP_NUM_THREADS'] = "4"
os.environ['MKL_NUM_THREADS'] = "4"
os.environ['NUMEXPR_NUM_THREADS'] = "4"
use_xgboost = True

xgb_search_space = [
    {'name': 'booster', 'type': 'cat', 'categories': ['gbtree', 'dart']},
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
    {'name': 'max_depth', 'type': 'int', 'lb': 1, 'ub': 3},
    {'name': 'n_jobs', 'type': 'cat', 'categories': [1]},
    # {'name': 'objective', 'type': 'cat', 'categories': ['rank:pairwise', 'reg:squarederror']},
    {'name': 'objective', 'type': 'cat', 'categories': ['rank:pairwise']},
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


class LGBMPairwiseRanker(LGBMRanker):

    def fit(self, X, y=None, group_id=None, **kwargs):
        group_id = np.array([X.shape[0]])
        return super().fit(X, y, group=group_id, **kwargs)

    def predict(self, X, **kwargs):
        return super().predict(X, **kwargs)


@notify
def simple_cv_task(fe=False, custom_fe=False, final_prediction=True):
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
        top_features = 50

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
                                     list(code_importance_dict.keys())[:top_features],
                                     only_new_features=True)
        # BaggingRegressor(XGBRegressor(**history_best_parameters[i]), n_jobs=-1).fit(X_all_k, Y_all_k).predict(X_all_k)
        score = np.mean(cross_val_score(
            # XGBRegressor(**history_best_parameters['A'][i]),
            XGBRegressor(n_estimators=100, max_depth=1, objective='rank:pairwise', n_jobs=1,
                         learning_rate=0.8),
            X_all_k, Y_all_k, cv=KFold(shuffle=True),
            scoring=make_scorer(kendalltau), n_jobs=-1))
        print('XGBRanker', score)
        scores.append(score)

        if final_prediction:
            # 最终预测
            # xgb = XGBRegressor(**history_best_parameters['A'][i])
            xgb = XGBRegressor(n_estimators=100, max_depth=1, objective='rank:pairwise', n_jobs=1,
                               learning_rate=0.8)
            test_data_np = np.array(test_arch_list)
            xgb.fit(X_all_k, Y_all_k)
            if fe:
                test_data_np = feature_append(ef, test_data_np,
                                              list(code_importance_dict.keys())[:top_features],
                                              only_new_features=True)
            prediction = xgb.predict(test_data_np)
            prediction_results.append(prediction)
    print('平均分', np.mean(scores))
    keys = list(test_data.keys())
    data_print(keys, prediction_results)


@notify
def ef_prediction():
    # 这个任务主要用来计算平均分数
    prediction_results = []
    ef = EvolutionaryForestRegressor(max_height=5, normalize=True, select='AutomaticLexicase',
                                     gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                     base_learner='Random-DT', verbose=True, n_process=1)
    for i in range(len(train_list[:])):
        print('Step', i)
        X_all_k, Y_all_k = np.array(arch_list_train), np.array(train_list[i])
        ef.lazy_init(X_all_k)
        with open(f'model_{i}.pkl', 'rb') as f:
            ef = dill.load(f)
        prediction = ef.predict(np.array(test_arch_list))
        prediction_results.append(prediction)
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
            新思路：
            1. 考虑采用多目标方式进行超参数优化
            """

            def parameters_to_xgb(parameters):
                return [(f'{id}', core_class(**parameter))
                        for id, parameter in enumerate(parameters)]

            search_space = xgb_search_space if use_xgboost else catboost_params
            best_parameter = sklearn_tuner(core_class, search_space, X_all_k, Y_all_k,
                                           metric=kendalltau,
                                           max_iter=16)
            print(best_parameter)
            model = VotingRegressor(parameters_to_xgb(best_parameter))
            # model = core_class(**best_parameter)
            print('XGBRanker', np.mean(cross_val_score(model,
                                                       X_all_k, Y_all_k,
                                                       scoring=make_scorer(kendalltau), n_jobs=-1)))
            xgb = model.fit(X_all_k, Y_all_k)
        else:
            """
            可以考虑的思路：
            1. Ensemble 目前无提升效果 线上分数：0.77526	
            2. RankSVM 离线效果不行
            3. 尝试一下RankDT
            
            * 需要产生更多Diverse的Base Learner，从而形成一个比较强的Ensemble
            * 单调性假设是否能增强模型性能？
            离线平均分 -0.00927655310621242
            很明显不能增强模型性能
            """
            temp_parameter = copy.deepcopy(history_best_parameters)
            # base = VotingRegressor(
            #     [
            #         (f'{xx}', XGBRegressor(**history_best_parameters[xx][i]))
            #         for xx in ['A', 'B', 'C', 'D']
            #     ],
            # )
            monotone_constraints = tuple([1 for _ in range(X_all_k.shape[1])])
            base = XGBRegressor(monotone_constraints=monotone_constraints, **history_best_parameters['A'][i])
            cv_prediction = cross_val_predict(base, X_all_k, Y_all_k, n_jobs=-1)
            mean_score = kendalltau(cv_prediction, Y_all_k)
            # score = cross_val_score(base, X_all_k, Y_all_k,
            #                         scoring=make_scorer(kendalltau))
            # mean_score = np.mean(score)
            scores.append(mean_score)
            print('XGBRanker', mean_score)
            xgb = base.fit(X_all_k, Y_all_k)
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

特征工程后的结果
任务 0.2975353535353536
任务 0.8600404040404042
任务 0.8837171717171719
任务 0.9477171717171717
任务 0.8792727272727274
任务 0.6746666666666667
任务 0.9096565656565657
任务 0.7925656565656566
总分 0.7806464646464647
"""


@notify
def joint_training_task():
    """
    多任务联合训练
    """

    x = arch_list_train.repeat(8, 0)
    y = np.concatenate(train_list)
    x = x @ np.random.randn(x.shape[1], x.shape[1])
    group_id = np.repeat(np.arange(8), len(arch_list_train))
    # np.concatenate([x, group_id.reshape(-1, 1)], axis=1)
    print(score_evaluation(XGBRanker(max_depth=1, n_jobs=1), x, y, group_id))


def score_evaluation(model, X_all_k, Y_all_k, qid_all):
    """
    交叉验证评估工具类
    此处交叉验证的思路为：将所有任务的数据合并在一起，通过GBDT进行联合训练
    很明显的一个问题是，不同任务存在冲突情况
    """
    scores = []
    for id in KFold(5, shuffle=True).split(X_all_k):
        train_id, test_id = id
        # QID代表了任务编号
        ranker = model.fit(X_all_k[train_id], Y_all_k[train_id], qid=qid_all[train_id])
        for id in np.unique(qid_all[test_id]):
            index = qid_all[test_id] == id
            score = kendalltau(Y_all_k[test_id][index], ranker.predict(X_all_k[test_id][index]))
            scores.append(score)
            print(id, score)
    return np.mean(np.array(scores))


if __name__ == '__main__':
    # tuning_task(tuning=False)
    tuning_task(tuning=True)
    # simple_cv_task(fe=True)
    # ef_prediction()
    # joint_training_task()
