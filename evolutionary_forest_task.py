# 基于演化森林构造特征
import dill
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance
from xgboost import XGBRegressor

from base_component.evolutionary_forest_plus import EvolutionaryForestRegressorPlus
from dataset_loader import *
from utils.notify_utils import notify

pairwise_xgb = XGBRegressor(n_estimators=100, max_depth=1, objective='rank:pairwise', n_jobs=1,
                            learning_rate=0.8)
#
# xgb_search_space = [
#     {'name': 'select', 'type': 'cat', 'categories': ['Tournament-7', 'AutomaticLexicase']},
# ]


ef = EvolutionaryForestRegressorPlus(max_height=8, normalize=False, select='Tournament-7',
                                     gene_num=20, boost_size=10, n_gen=200, n_pop=200,
                                     cross_pb=0.9, mutation_pb=0.1,
                                     # base_learner='Random-DT',
                                     base_learner=pairwise_xgb,
                                     verbose=True, n_process=64,
                                     original_features=False)


@notify
def feature_construction():
    # 所有数据联合训练
    # X_all_k, Y_all_k = np.array(arch_list_train).repeat(8, axis=0), \
    #                    np.concatenate([train_list[i] for i in range(len(train_list))])
    # X_all_k = np.concatenate([X_all_k, np.arange(0, len(X_all_k)).reshape(-1, 1) // len(arch_list_train)], axis=1)
    for i in range(0, 8):
        X_all_k, Y_all_k = arch_list_train, np.array(train_list[i])
        ef.fit(X_all_k, Y_all_k)

        feature_importance_dict = get_feature_importance(ef)
        plot_feature_importance(feature_importance_dict)

        with open(f'model_{i}.pkl', 'wb') as f:
            if hasattr(ef, 'pool'):
                del ef.pool
            dill.dump(ef, f)


if __name__ == '__main__':
    feature_construction()
