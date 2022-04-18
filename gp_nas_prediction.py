import numpy as np
import scipy.stats
from paddleslim.nas import GPNAS

from dataset_loader import train_list, arch_list_train

gp_list = []


def task(c_flag=2, m_flag=2):
    for i in range(len(train_list[:])):
        # 每个任务有该任务专属的gpnas预测器
        gp_list.append(GPNAS(c_flag, m_flag))

    train_num = 400
    all_score = []
    for i in range(len(train_list[:])):
        # 划分训练及测试集
        X_all_k, Y_all_k = np.array(arch_list_train), np.array(train_list[i])
        X_train_k, Y_train_k, X_test_k, Y_test_k = X_all_k[0:train_num:1], Y_all_k[0:train_num:1], \
                                                   X_all_k[train_num::1], Y_all_k[train_num::1]
        # 初始该任务的gpnas预测器参数
        gp_list[i].get_initial_mean(X_train_k[0::2], Y_train_k[0::2])
        init_cov = gp_list[i].get_initial_cov(X_train_k)
        # 更新（训练）gpnas预测器超参数
        gp_list[i].get_posterior_mean(X_train_k[1::2], Y_train_k[1::2])

        # 基于测试评估预测误差
        # error_list_gp = np.array(Y_test_k.reshape(len(Y_test_k), 1) - gp_list[i].get_predict(X_test_k))
        # error_list_gp_j = np.array(Y_test_k.reshape(len(Y_test_k), 1) -
        #                            gp_list[i].get_predict_jiont(X_test_k, X_train_k[::1], Y_train_k[::1]))
        # print('AVE mean gp :', np.mean(abs(np.divide(error_list_gp, Y_test_k.reshape(len(Y_test_k), 1)))))
        # print('AVE mean gp joint :', np.mean(abs(np.divide(error_list_gp_j, Y_test_k.reshape(len(Y_test_k), 1)))))
        y_predict = gp_list[i].get_predict_jiont(X_test_k, X_train_k[::1], Y_train_k[::1])
        # 基于测试集评估预测的Kendalltau
        kendalltau = scipy.stats.stats.kendalltau(y_predict, Y_test_k)
        all_score.append(kendalltau)
        print('Kendalltau:', kendalltau)
    print(c_flag, m_flag, np.mean(all_score))


class GPNASRegressor(GPNAS):
    def fit(self, X_train_k, Y_train_k):
        self.X_train_k = X_train_k
        self.Y_train_k = Y_train_k
        self.get_initial_mean(X_train_k[0::2], Y_train_k[0::2])
        self.get_initial_cov(X_train_k)
        # 更新（训练）gpnas预测器超参数
        self.get_posterior_mean(X_train_k[1::2], Y_train_k[1::2])
        return self

    def predict(self, X):
        return self.get_predict_jiont(X, self.X_train_k[::1], self.Y_train_k[::1])


if __name__ == '__main__':
    for a in [1, 2]:
        for b in [1, 2]:
            task(a, b)
