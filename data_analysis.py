# 查看一下数据本身的相关性
import matplotlib.pyplot as plt
from evolutionary_forest.forest import spearman
from seaborn import heatmap

from dataset_loader import *

data = np.zeros((len(name_list), len(name_list)))
for idx, name in enumerate(name_list):
    for idx_b, name_b in enumerate(name_list):
        print(name, name_b, spearman(train_list[idx], train_list[idx_b]))
        data[idx, idx_b] = spearman(train_list[idx], train_list[idx_b])
heatmap(data)
plt.title('Correlation Matrix')
plt.show()
