# -*- coding:utf8 -*-
# @TIME     : 2019/3/6 15:39
# @Author   : SuHao
# @File     : Su_KNN.py

import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 21:58:57 2019

@author: Hao
"""

import numpy as np
import copy
import matplotlib.pyplot as plt


def display(x):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=True,
        sharey=True, )
    ax = ax.flatten()
    for i in range(2):
        img = x.reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def distsort(a, k):
    '''
    \param: a:array
    \param: k:排序的个数
    '''
    if a.shape[1] != 2:
        return
    x = copy.copy(a)
    if k == 0:
        k = len(x)
    y = x[0:k, :]  # 先选取前k个存在队列里，之后要进行排序

    for i in range(k):
        for j in range(k - 1 - i):
            if y[j, 0] >= y[j + 1, 0]:
                y[[j, j + 1], :] = y[[j + 1, j], :]  # 冒泡排序

    for i in range(np.shape(x)[0] - k):
        for j in range(k - 1):
            if x[k + i, 0] >= y[k - 1, 0]:
                break
            elif x[k + i, 0] < y[0, 0]:
                y = np.insert(y, 0, x[k + i, :], axis=0)
                break
            elif x[k + i, 0] < y[k - 1 - j, 0] and x[k + i, 0] >= y[k - 2 - j, 0]:
                y = np.insert(y, k - 1 - j, x[k + i, :], axis=0)
                break
    return y[:k, :]


def force_research(dist_form, train_X, train_Y, test_x, k, kind):
    '''
    :param dist_form: lambda表达式，计算距离的公式
    :param train_X: 训练数据
    :param train_Y: 训练数据标签
    :param test_x: 测试数据，只有一个点
    :param test_y: 测试数据的标签
    :param k: 近邻数
    :param kind: 排序方法，可选参数：'quicksort', 'mergesort', 'heapsort', 'stable'
    :return: 返回对应近邻点的标签
    '''
    if dist_form is None:
        def dist_form(x, y): return np.sqrt(np.sum((x - y)**2))

    dist = np.zeros((np.shape(train_X)[1], ))
    for index, x in enumerate(train_X.T):
        dist[index] = dist_form(x, test_x)

    # dist = np.sqrt(np.sum((np.tile(test_x,(np.shape(train_X)[1],1)).T - train_X)**2, axis=0))

    # params = {'a': dist, 'kind': kind}
    # a = train_Y[np.argsort(**params)]
    # a[0:k]

    table = np.array([dist, train_Y]).T
    neighbor = distsort(table, k)
    # print(neighbor[0:k, 0])
    return neighbor[0:k, 1]


class classifier:

    def __init__(self, research, dist):
        '''
        :param research: 搜索方法，两种选择：force和kdtree
        :param dist: lambda表达式，计算距离
        '''
        # self.research = research
        pass

    def train(self, train_X, train_Y, prior):
        '''
        :param train_X: training data, which size is M*N, where M is number of attribution and N is num of sample.
        :param trian_Y: corresponding label, string or value
        :param prior: 先验概率，可以指定
        :return:
        '''
        self.train_X = train_X
        self.train_Y = train_Y
        if prior is None:
            self.prior = np.bincount(train_Y) / len(train_Y)
        else:
            self.prior = prior
        self.label_num = np.shape(self.prior)[0]  # label必须从0开始，否则此会出错

    def predict(self, dist_form, k, test_X, kind):
        predict_result = np.zeros((np.shape(test_X)[1], ))
        predict_prob = np.zeros((self.label_num, np.shape(test_X)[1], ))
        for index, test_x in enumerate(test_X.T):
            # display(test_x)
            neighbor = force_research(
                train_X=self.train_X,
                train_Y=self.train_Y,
                dist_form=dist_form,
                test_x=test_x,
                k=k,
                kind=kind)
            neighbor = neighbor.astype(np.int8)
            predict_prob[:, index] = np.bincount(
                neighbor, minlength=self.label_num) / k
            # predict_prob[index, :] = np.bincount(neighbor, minlength=self.label_num) * self.prior / k
            predict_result[index] = np.argmax(predict_prob[:, index])

        return predict_result, predict_prob

    def test(self, test_X, test_Y, kind, k, dist_form):
        predict_result, predict_prob = self.predict(dist_form, k, test_X, kind)
        accurate = np.sum(predict_result == test_Y) / np.shape(test_Y)[0]
        return accurate
