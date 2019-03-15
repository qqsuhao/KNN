# -*- coding:utf8 -*-
# @TIME     : 2019/3/5 9:34
# @Author   : SuHao
# @File     : Su_kdtree.py

'''
This is Su Hao's Kdtree programm, refering the KDTree open source code.
'''

from __future__ import print_function

import heapq
import itertools
import operator
import math
from collections import deque
from functools import wraps


class Node(object):
    def __init__(self, data=None, left=None, right=None):
        # 此处 left和right应该是class Node类型
        self.data = data
        self.left = left
        self.right = right

    @property  # 把成员函数变成属性
    def is_leaf(self):
        '''Returns True if a Node has no subnodes.'''
        return (not self.data) or \
               (all(not bool(c) for c, p in self.children))

    def preorder(self):
        if not self:  # 递归终止条件
            return

        yield self
        '''
        二叉树的先序遍历：使用了递归与生成器结合的方法，效果特别棒。
        迭代器：先序遍历是一个比较长的序列，可以用生成器来产生。
        递归思想：从根结点先序遍历，先要对子结点进行先序遍历。
        比较特殊的地方在于，在迭代器中调用迭代器。根节点的迭代器要靠子结点的迭代器来构造。
        直到叶子结点，其迭代器里只有一个元素就是叶子自己。
        这里迭代器的作用有些像在C++里使用队列来存放遍历结果
        '''
        if self.left:
            for x in self.left.preorder():
                yield x

        if self.right:
            for x in self.right.preorder():
                yield x

    def inorder(self):
        if not self:
            return

        if self.left:
            for x in self.left.inorder():
                yield x

        yield self

        if self.right:
            for x in self.right.inroder():
                yield x

    def postorder(self):
        if not self:
            return

        if self.left:
            for x in self.left.postorder():
                yield x

        if self.right:
            for x in self.right.postorder():
                yield x

        yield self

    @property
    def children(self):
        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1
    # 此处迭代器的元素为一个数对，因此在使用循环访问迭代器元素时，要使用一对儿变量访问

    def set_child(self, index, child):
        if index == 0:
            self.left = child
        else:
            self.right = child

    def height(self):
        min_height = int(bool(self))
        return max([min_height] + [c.height() + 1 for c, p in self.children])
        '''
        此处迭代器的元素为一个数对，因此在使用循环访问迭代器元素时，要使用一对儿变量访问
        使用递归计算树的高度；还用了个max([]+[])，真是绝了，打死我也写不出来这种code。
        '''

    def get_child_pos(self, child):
        for c, pos in self.children:
            if child == c:
                return pos

    def __repr__(self):
        return '<%(cls)s - %(data)s>' %\
            dict(cls=self.__class__.__name__, data=repr(self.data))

    def __nonzero__(self):
        return self.data is not None

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

    def __hash__(self):
        return id(self)


def require_axis(f):
    """ Check if the object of the function has axis and sel_axis members """

    @wraps(f)
    def _wrapper(self, *args, **kwargs):
        if None in (self.axis, self.sel_axis):
            raise ValueError('%(func_name) requires the node %(node)s '
                             'to have an axis and a sel_axis function' %
                             dict(func_name=f.__name__, node=repr(self)))

        return f(self, *args, **kwargs)

    return _wrapper


class KDNode(Node):
    def __init__(self, data=None, left=None, right=None, axis=None,
                 sel_axis=None, dimensions=None):
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions

    @require_axis
    def add(self, point):
        '''
        在当前节点判断如果为空，则data赋值为point；
        否则，将point指定维度的坐标值与data相比，
        小于则往二叉树的左边走；大于则往二叉树的右边走；
        往左边走的时候如果是空结点，就调用subnode函数创建子结点，并且poin幅值给这个子节点
        如果不是空结点，则将这个子结点视为当前结点。
        可以预料之后的行为与上述相同.
        然后不停的循环。
        '''
        current = self
        while True:
            check_dimensionality([point], dimensions=current.dimensions)

            if current.data is None:
                current.data = point
                return current

            if point[current.axis] < current.data[current.axis]:
                if current.left is None:
                    current.left = current.creat_subnode(point)
                    return current.left
                else:
                    current = current.left
            else:
                if current.right is None:
                    current.right = current.create_subnode(point)
                    return current.right
                else:
                    current = current.right

    @require_axis
    def creat_subnode(self, data):
        """ Creates a subnode for the current node
        在类成员函数中创建自己的实例，需要用self.__class__();
        相当于在类的外面使用Node()来创建类的实例 """
        return self.__class__(data,
                              axis=self.sel_axis(self.axis),
                              sel_axis=self.sel_axis,
                              dimensions=self.dimensions)


    @require_axis
    def find_replacement(self):
        """ Finds a replacement for the current node

        The replacement is returned as a
        (replacement-node, replacements-parent-node) tuple """

        if self.right:
            child, parent = self.right.extreme_child(min, self.axis)
        else:
            child, parent = self.left.extreme_child(max, self.axis)

        return (child, parent if parent is not None else self)


    def should_remove(self, point, node):
        """ checks if self's point (and maybe identity) matches """
        if not self.data == point:
            return False

        return (node is None) or (node is self)


    @require_axis
    def remove(self, point, node=None):
        """ Removes the node with the given point from the tree

        Returns the new root node of the (sub)tree.

        If there are multiple points matching "point", only one is removed. The
        optional "node" parameter is used for checking the identity, once the
        removeal candidate is decided."""

        # Recursion has reached an empty leaf node, nothing here to delete
        if not self:
            return

        # Recursion has reached the node to be deleted
        if self.should_remove(point, node):
            return self._remove(point)

        # Remove direct subnode
        if self.left and self.left.should_remove(point, node):
            self.left = self.left._remove(point)

        elif self.right and self.right.should_remove(point, node):
            self.right = self.right._remove(point)

        # Recurse to subtrees
        if point[self.axis] <= self.data[self.axis]:
            if self.left:
                self.left = self.left.remove(point, node)

        if point[self.axis] >= self.data[self.axis]:
            if self.right:
                self.right = self.right.remove(point, node)

        return self


    @require_axis
    def _remove(self, point):
        # we have reached the node to be deleted here

        # deleting a leaf node is trivial
        if self.is_leaf:
            self.data = None
            return self

        # we have to delete a non-leaf node here

        # find a replacement for the node (will be the new subtree-root)
        root, max_p = self.find_replacement()

        # self and root swap positions
        tmp_l, tmp_r = self.left, self.right
        self.left, self.right = root.left, root.right
        root.left, root.right = tmp_l if tmp_l is not root else self, tmp_r if tmp_r is not root else self
        self.axis, root.axis = root.axis, self.axis

        # Special-case if we have not chosen a direct child as the replacement
        if max_p is not self:
            pos = max_p.get_child_pos(root)
            max_p.set_child(pos, self)
            max_p.remove(point, self)

        else:
            root.remove(point, self)

        return root


    @property
    def is_balanced(self):
        """ Returns True if the (sub)tree is balanced

        The tree is balanced if the heights of both subtrees differ at most by
        1 """

        left_height = self.left.height() if self.left else 0
        right_height = self.right.height() if self.right else 0

        if abs(left_height - right_height) > 1:
            return False

        return all(c.is_balanced for c, _ in self.children)


    def rebalance(self):
        """
        Returns the (possibly new) root of the rebalanced tree
        """

        return create([x.data for x in self.inorder()])


    def axis_dist(self, point, axis):
        """
        Squared distance at the given axis between
        the current Node and the given point
        """
        return math.pow(self.data[axis] - point[axis], 2)


    def dist(self, point):
        """
        Squared distance between the current Node
        and the given point
        """
        r = range(self.dimensions)
        return sum([self.axis_dist(point, i) for i in r])


    def search_knn(self, point, k, dist=None):
        """ Return the k nearest neighbors of point and their distances

        point must be an actual point, not a node.

        k is the number of results to return. The actual results can be less
        (if there aren't more nodes to return) or more in case of equal
        distances.

        dist is a distance function, expecting two points and returning a
        distance value. Distance values can be any compareable type.

        The result is an ordered list of (node, distance) tuples.
        """

        if k < 1:
            raise ValueError("k must be greater than 0.")

        if dist is None:
            get_dist = lambda n: n.dist(point)
        else:
            get_dist = lambda n: dist(n.data, point)

        results = []

        self._search_node(point, k, results, get_dist, itertools.count())

        # We sort the final result by the distance in the tuple
        # (<KdNode>, distance).
        return [(node, -d) for d, _, node in sorted(results, reverse=True)]


    def _search_node(self, point, k, results, get_dist, counter):
        if not self:
            return

        nodeDist = get_dist(self)

        # Add current node to the priority queue if it closer than
        # at least one point in the queue.
        #
        # If the heap is at its capacity, we need to check if the
        # current node is closer than the current farthest node, and if
        # so, replace it.
        item = (-nodeDist, next(counter), self)
        if len(results) >= k:
            if -nodeDist > results[0][0]:
                heapq.heapreplace(results, item)
        else:
            heapq.heappush(results, item)
        # get the splitting plane
        split_plane = self.data[self.axis]
        # get the squared distance between the point and the splitting plane
        # (squared since all distances are squared).
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist * plane_dist

        # Search the side of the splitting plane that the point is in
        if point[self.axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point, k, results, get_dist, counter)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, get_dist, counter)

        # Search the other side of the splitting plane if it may contain
        # points closer than the farthest point in the current results.
        if -plane_dist2 > results[0][0] or len(results) < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, get_dist,
                                            counter)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, get_dist,
                                           counter)


    @require_axis
    def search_nn(self, point, dist=None):
        """
        Search the nearest node of the given point

        point must be an actual point, not a node. The nearest node to the
        point is returned. If a location of an actual node is used, the Node
        with this location will be returned (not its neighbor).

        dist is a distance function, expecting two points and returning a
        distance value. Distance values can be any compareable type.

        The result is a (node, distance) tuple.
        """

        return next(iter(self.search_knn(point, 1, dist)), None)


    def _search_nn_dist(self, point, dist, results, get_dist):
        if not self:
            return

        nodeDist = get_dist(self)

        if nodeDist < dist:
            results.append(self.data)

        # get the splitting plane
        split_plane = self.data[self.axis]

        # Search the side of the splitting plane that the point is in
        if point[self.axis] <= split_plane + dist:
            if self.left is not None:
                self.left._search_nn_dist(point, dist, results, get_dist)
        if point[self.axis] >= split_plane - dist:
            if self.right is not None:
                self.right._search_nn_dist(point, dist, results, get_dist)


    @require_axis
    def search_nn_dist(self, point, distance, best=None):
        """
        Search the n nearest nodes of the given point which are within given
        distance

        point must be a location, not a node. A list containing the n nearest
        nodes to the point within the distance will be returned.
        """

        results = []
        get_dist = lambda n: n.dist(point)

        self._search_nn_dist(point, distance, results, get_dist)
        return results


    @require_axis
    def is_valid(self):
        """ Checks recursively if the tree is valid

        It is valid if each node splits correctly """

        if not self:
            return True

        if self.left and self.data[self.axis] < self.left.data[self.axis]:
            return False

        if self.right and self.data[self.axis] > self.right.data[self.axis]:
            return False

        return all(c.is_valid() for c, _ in self.children) or self.is_leaf


    def extreme_child(self, sel_func, axis):
        """ Returns a child of the subtree and its parent

        The child is selected by sel_func which is either min or max
        (or a different function with similar semantics). """

        max_key = lambda child_parent: child_parent[0].data[axis]


        # we don't know our parent, so we include None
        me = [(self, None)] if self else []

        child_max = [c.extreme_child(sel_func, axis) for c, _ in self.children]
        # insert self for unknown parents
        child_max = [(c, p if p is not None else self) for c, p in child_max]

        candidates =  me + child_max

        if not candidates:
            return None, None

        return sel_func(candidates, key=max_key)



def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
    """ Creates a kd-tree from a list of points

    All points in the list must be of the same dimensionality.

    If no point_list is given, an empty tree is created. The number of
    dimensions has to be given instead.

    If both a point_list and dimensions are given, the numbers must agree.

    Axis is the axis on which the root-node should split.

    sel_axis(axis) is used when creating subnodes of a node. It receives the
    axis of the parent node and returns the axis of the child node. """

    if not point_list and not dimensions:
        raise ValueError('either point_list or dimensions must be provided')

    elif point_list:
        dimensions = check_dimensionality(point_list, dimensions)

    # by default cycle through the axis
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis+1) % dimensions)

    if not point_list:
        return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)

    # Sort point list and choose median as pivot element
    point_list = list(point_list)
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2

    loc   = point_list[median]
    left  = create(point_list[:median], dimensions, sel_axis(axis))
    right = create(point_list[median + 1:], dimensions, sel_axis(axis))
    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)


def check_dimensionality(point_list, dimensions=None):
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError('All Points in the point_list must have the same dimensionality')

    return dimensions



def level_order(tree, include_all=False):
    """ Returns an iterator over the tree in level-order

    If include_all is set to True, empty parts of the tree are filled
    with dummy entries and the iterator becomes infinite. """

    q = deque()
    q.append(tree)
    while q:
        node = q.popleft()
        yield node

        if include_all or node.left:
            q.append(node.left or node.__class__())

        if include_all or node.right:
            q.append(node.right or node.__class__())



def visualize(tree, max_level=100, node_width=10, left_padding=5):
    """ Prints the tree to stdout """

    height = min(max_level, tree.height()-1)
    max_width = pow(2, height)

    per_level = 1
    in_level  = 0
    level     = 0

    for node in level_order(tree, include_all=True):

        if in_level == 0:
            print()
            print()
            print(' '*left_padding, end=' ')

        width = int(max_width*node_width/per_level)

        node_str = (str(node.data) if node else '').center(width)
        print(node_str, end=' ')

        in_level += 1

        if in_level == per_level:
            in_level   = 0
            per_level *= 2
            level     += 1

        if level > height:
            break

    print()
    print()
