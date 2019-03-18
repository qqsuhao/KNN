# -*- coding:utf8 -*-
# @TIME     : 2019/3/15 22:48
# @Author   : SuHao
# @File     : test4.py

import Su_kdtree as kd
import numpy as np

np.random.seed(0)
a = np.random.randint(0,50,(10,2))
a = a.tolist()
sel_axis=lambda x: (x+1) % 2
tree = kd.create(a, dimensions=2, axis=0, sel_axis=sel_axis)
kd.visualize(tree)


c = np.append(a, np.random.randint(0,50,(10,1)),axis=1)
c = c.tolist()
tree2 = kd.create(c, dimensions=3, axis=0, sel_axis=lambda x: (x+1) % 2)
b = tree2.search_knn([2,3,0],3)
kd.visualize(tree2)