# -*- coding:utf8 -*-
# @TIME     : 2019/3/13 10:42
# @Author   : SuHao
# @File     : test2.py

import Su_KNN as KNN
import os
import struct
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier

'''
minist数据集中的每个样本数据为784*1的一个数组，表示一张28*28的图片；
训练数据有60000个，测试数据有10000个
'''


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


train_X, train_Y = load_mnist('.', kind='train')
test_X, test_Y = load_mnist('.', kind='t10k')

train_X = train_X.astype('float64')
test_X = test_X.astype('float64')

N = 10
start1 = time.perf_counter()
neigh = KNeighborsClassifier(
    n_neighbors=5,
    algorithm='brute',
    metric='euclidean')
neigh.fit(train_X, train_Y)
Pe1 = neigh.score(test_X[0:N, :], test_Y[0:N])
# dist1, result1 = neigh.kneighbors(X=test_X[1:2, :], n_neighbors=5, return_distance=True)
# print(dist1)
end1 = time.perf_counter()
print(end1 - start1, Pe1)


params = {'dist_form': None,
          'test_X': test_X.T[:, 0:N],
          'test_Y': test_Y[0:N],
          'k': 5,
          'kind': 'heapsort'}
start = time.perf_counter()
test = KNN.classifier('force', None)
test.train(train_X.T, train_Y, None)
pe = test.test(**params)
# predict_result, predict_prob = test.predict(dist_form=None, k=5, test_X=test_X[:, 0:N], kind='heapsort')
end = time.perf_counter()
print(end - start, pe)
