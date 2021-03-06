import dill
from evolutionary_forest.forest import EvolutionaryForestRegressor
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance

from dataset_loader import arch_list_train

ef = EvolutionaryForestRegressor(max_height=5, normalize=True, select='AutomaticLexicase',
                                 gene_num=10, boost_size=100, n_gen=20, n_pop=200, cross_pb=1,
                                 base_learner='Random-DT', verbose=True, n_process=1)
ef.lazy_init(arch_list_train)

for i in range(8):
    with open(f'model_{i}.pkl', 'rb') as f:
        ef = dill.load(f)
    code_importance_dict = get_feature_importance(ef)
    plot_feature_importance(code_importance_dict)
