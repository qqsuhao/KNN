# -*- coding:utf8 -*-
# @TIME     : 2019/3/16 12:01
# @Author   : SuHao
# @File     : dist_formular.py

from sklearn.neighbors import DistanceMetric
import numpy as np

'''
行数为点的个数，列数为点的维度
'''


def euclidean(x, y):
    result = DistanceMetric.get_metric('euclidean')
    return result.pairwise(x, y)


def manhattan(x, y):
    result = DistanceMetric.get_metric('manhattan')
    return result.pairwise(x, y)


def chebyshev(x, y):
    result = DistanceMetric.get_metric('chebyshev')
    return result.pairwise(x, y)


def minkowski(x, y, p):
    result = DistanceMetric.get_metric('minkowski', p)
    return result.pairwise(x, y)


def wminkowski(x, y, p ,w):
    result = DistanceMetric.get_metric('wminkowski', p, w)
    return result.pairwise(x, y)


def seuclidean(x, y, V):
    result = DistanceMetric.get_metric('seuclidean', V)
    return result.pairwise(x, y)


def manhalanobis(x, y, V):
    result = DistanceMetric.get_metric('manhalanobis', V)
    return result.pairwise(x, y)


'''以下距离公式参考
https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
'''
def correlation(x, y):
    def sigle(a, b):
        np.
