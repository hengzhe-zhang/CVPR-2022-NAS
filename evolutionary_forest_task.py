# 基于演化森林构造特征
import dill
from catboost import Pool
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance
from sklearn.datasets import load_boston
from xgboost import XGBRegressor

from base_component.evolutionary_forest_plus import EvolutionaryForestRegressorPlus
from dataset_loader import *
from utils.notify_utils import notify
from xgb_prediction import CatBoostPairwiseRanker

pairwise_xgb = CatBoostPairwiseRanker(n_estimators=50, max_depth=1, loss_function='PairLogit',
                                      thread_count=1, learning_rate=0.9, verbose=False)

ef = EvolutionaryForestRegressorPlus(max_height=3, normalize=False, select='Tournament-7',
                                     gene_num=10, boost_size=10, n_gen=10, n_pop=100,
                                     cross_pb=0.9, mutation_pb=0.1,
                                     # base_learner='Random-DT',
                                     base_learner=pairwise_xgb,
                                     verbose=True, n_process=58,
                                     original_features=True)


@notify
def feature_construction():
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
    # X, y = load_boston(return_X_y=True)
    # pairwise_xgb.fit(X, y)
    # print(pairwise_xgb.get_feature_importance(Pool(X)))
