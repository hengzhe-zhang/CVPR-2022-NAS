import json

# 查看训练集
from collections import defaultdict

import numpy as np

with open('CVPR_2022_NAS_Track2_train.json', 'r') as f:
    train_data = json.load(f)
print('train_num:', len(train_data.keys()))

# 查看测试集
with open('CVPR_2022_NAS_Track2_test.json', 'r') as f:
    test_data = json.load(f)


# 处理训练数据
def convert_X(arch_str):
    temp_arch = []
    for id, elm in enumerate(arch_str):
        # 长度编码
        if elm == 'l':
            temp_arch.append(12)
        elif elm == 'j':
            temp_arch.append(10)
        elif elm == 'k':
            temp_arch.append(11)
        else:
            temp_arch.append(int(elm))
    return (temp_arch)


# 处理训练集
train_list = [[], [], [], [], [], [], [], []]
arch_list_train = []
name_list = ['cplfw_rank', 'market1501_rank', 'dukemtmc_rank', 'msmt17_rank', 'veri_rank', 'vehicleid_rank',
             'veriwild_rank', 'sop_rank']
# 训练数据准备
for key in train_data.keys():
    # 预测分数
    for idx, name in enumerate(name_list):
        train_list[idx].append(train_data[key][name])
    # 训练数据
    arch_list_train.append(convert_X(train_data[key]['arch']))
arch_list_train = np.array(arch_list_train).astype(np.float32)
index = np.array([0] + [i for i in range(1, arch_list_train.shape[1], 3)]
                 + [i for i in range(2, arch_list_train.shape[1], 3)])
# 被删除的数据
# arch_list_train[:,np.array([i for i in range(3, arch_list_train.shape[1], 3)])]
arch_list_train = arch_list_train[:, index]
heads = arch_list_train[:, 1::2]
ratios = arch_list_train[:, 2::2]
heads[heads == 1] = 12
heads[heads == 2] = 11
heads[heads == 3] = 10
ratios[ratios == 1] = 4
ratios[ratios == 3] = 3
ratios[ratios == 2] = 3.5

# 处理测试集数据
test_arch_list = []
for key in test_data.keys():
    test_arch = convert_X(test_data[key]['arch'])
    test_arch_list.append(test_arch)
test_arch_list = np.array(test_arch_list).astype(np.float32)
test_arch_list = test_arch_list[:, index]
heads = test_arch_list[:, 1::2]
ratios = test_arch_list[:, 2::2]
heads[heads == 1] = 12
heads[heads == 2] = 11
heads[heads == 3] = 10
ratios[ratios == 1] = 4
ratios[ratios == 3] = 3
ratios[ratios == 2] = 3.5

if __name__ == '__main__':
    dt = defaultdict(int)
    for i, x in enumerate(train_list[0]):
        for j, y in enumerate(train_list[0]):
            if j > i:
                if x > y:
                    dt[x] += 1
                    dt[y] -= 1
    print(dt)
