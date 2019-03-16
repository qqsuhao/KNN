# -*- coding:utf8 -*-
# @TIME     : 2019/3/15 22:48
# @Author   : SuHao
# @File     : test4.py

import Su_kdtree as kd
import numpy as np

np.random.seed(0)
a = np.random.randint(0,50,(10,2))
a = list(a)
tree = kd.create(a, dimensions=2, axis=0, sel_axis=None)
kd.visualize(tree)
b, c = tree.find_replacement()
# x2 = kd.level_order(tree)
