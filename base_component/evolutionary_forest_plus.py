import operator
from functools import partial

from deap import gp
from evolutionary_forest.forest import EvolutionaryForestRegressor, spearman
from evolutionary_forest.multigene_gp import *


def cxOnePoint_multiple_all_gene(ind1: MultipleGeneGP, ind2: MultipleGeneGP, probability):
    for a, b in zip(ind1.gene, ind2.gene):
        if random.random() < probability:
            cxOnePoint(a, b)
    return ind1, ind2


def mutate_all_gene(individual: MultipleGeneGP, expr, pset, probability):
    if random.random() < probability:
        mutUniform(individual.weight_select(), expr, pset)
    return individual,


class EvolutionaryForestRegressorPlus(EvolutionaryForestRegressor):
    """
    魔改之后的演化森林
    """

    def lazy_init(self, x):
        super().lazy_init(x)

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=0, max_=8)
        self.toolbox.register("mate", partial(cxOnePoint_multiple_all_gene, probability=self.cross_pb))
        self.toolbox.register("mutate", partial(mutate_all_gene,
                                                expr=self.toolbox.expr_mut, pset=self.pset,
                                                probability=self.mutation_pb))
        self.toolbox.decorate("mate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                                max_value=self.max_height))
        self.toolbox.decorate("mutate", staticLimit_multiple_gene(key=operator.attrgetter("height"),
                                                                  max_value=self.max_height))
        self.cross_pb = 1
        self.mutation_pb = 1

    def calculate_fitness_value(self, individual, Y, y_pred):
        # 评估函数定义
        if np.isnan(spearman(Y, y_pred)):
            return 1,
        return -1 * spearman(Y, y_pred),
