# -*- coding:utf8 -*-
# @TIME     : 2019/3/5 9:34
# @Author   : SuHao
# @File     : Su_kdtree.py

import numpy as np
import dist_formular as dist

np.random.seed(0)
a = np.random.randint(0,20,(1,3))
np.random.seed(1)
b = np.random.randint(0,20,(2,3))

c = map(lambda x,y: np.sqrt(np.sum((x-y)**2)), a, b)
print(list(c))

print(dist.euclidean(a,b))

