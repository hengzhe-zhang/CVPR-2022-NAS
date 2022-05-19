# 查看所有变量的分布范围
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dataset_loader import arch_list_train, test_arch_list

sns.set(style='whitegrid')

_, axes = plt.subplots(2, len(pd.DataFrame(arch_list_train.T)), figsize=(50, 6))
# Get the columns which differ a lot between test and train
for id, a, b, ax in zip(range(0, len(arch_list_train)), pd.DataFrame(arch_list_train.T).iterrows(),
                        pd.DataFrame(test_arch_list.T).iterrows(),
                        axes.T):
    a[1].hist(ax=ax[0], color='b')
    b[1].hist(ax=ax[1], color='g')
plt.tight_layout()
plt.show()
