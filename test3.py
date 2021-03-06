# -*- coding:utf8 -*-
# @TIME     : 2019/3/14 14:09
# @Author   : SuHao
# @File     : test3.py

import pandas as pd
import Su_KNN as KNN
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv(r"glass.txt", header=None, delimiter=',')
data = df.values

for index, i in enumerate(data[:, 10]):
    if i == 1:
        data[index, 10] = 0
    elif i == 2:
        data[index, 10] = 1
    elif i == 3:
        data[index, 10] = 2
    elif i == 5:
        data[index, 10] = 3
    elif i == 6:
        data[index, 10] = 4
    elif i == 7:
        data[index, 10] = 5

choice = [17,66,89,134,148,158,165,171,179,181,199,212]
# choice = np.random.randint(130,210,(20,))
test = data[choice, :]
train = np.delete(data, choice, 0)

train_X = train[:, 1:9].T
train_Y = train[:, 10].astype(np.int8)
test_X = test[:, 1:9].T
test_Y = test[:, 10].astype(np.int8)



start = time.perf_counter()
KK = KNN.classifier(research='kdtree', dist_form='euclidean', k=3)
KK.train(train_X, train_Y, None)
pe = KK.test(test_X, test_Y)
predict_result, predict_prob, neighbor_dist = KK.predict(test_X)
print(neighbor_dist)
end = time.perf_counter()
print(end - start, pe)


start1 = time.perf_counter()
neigh = KNeighborsClassifier(n_neighbors=3, algorithm='kd_tree', metric='euclidean')
neigh.fit(train_X.T, train_Y)
Pe1 = neigh.score(test_X.T, test_Y)
dist1, result1 = neigh.kneighbors(X=test_X.T, n_neighbors=3, return_distance=True)
a = neigh.predict(test_X.T)
print(a)
end1 = time.perf_counter()
print(end1 - start1, Pe1)