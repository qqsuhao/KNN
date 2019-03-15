# -*- coding:utf8 -*-
# @TIME     : 2019/3/15 16:57
# @Author   : SuHao
# @File     : kdtree总结.py


'''
class Node():
    def __init__(self, data, left, right):

    @ property
    def is_leaf(self):
        通过children函数判断该节点是否有子节点，如果没有子节点，则认为是叶子节点。
        通过@property修饰器，将这个函数等同为一个类的属性

    @ property
    def children(self):
        返回一个迭代器，其中存放键值对（左节点对应值为0，右节点为1）
        对于二叉树来说。这个迭代器最大长度就是2

    def set_child(self, index, child):
        设置子结点。如果index为0，则将child设置为左子节点，否则为右子结点。
        child应该也是Node对象

    def height(self):
        采用递归的思想：这个函数计算从当前节点往下的高度，
        等于1+max（左子结点往下的高度，右子结点往下的高度）。

    def get_child_pos(self, child):
        输入节点child，通过children迭代器判断是左子节点还是右子结点
        返回0表示左节点，1表示右节点

    def preorder(self):
        二叉树的先序遍历：使用了递归与生成器结合的方法，效果特别棒。
        迭代器：先序遍历是一个比较长的序列，可以用生成器来产生。
        递归思想：从根结点先序遍历，先要对子结点进行先序遍历。
        比较特殊的地方在于，在迭代器中调用迭代器。根节点的迭代器要靠子结点的迭代器来构造。
        直到叶子结点，其迭代器里只有一个元素就是叶子自己。
        这里迭代器的作用有些像在C++里使用队列来存放遍历结果

    def inorder(self):

    def postorder(self):
'''


'''
class KDNode(Node):
    def __init__(self, data, left, right, axis, sel_axis, dimension):
        重写父类构造函数；
        axis：从数据的哪一维度开始构建kdtree；
        sel_axis: 默认kdtree每次向下一层，被切割的维度+1，但是也可以通过这个参数更改策略
                  这个参数是一个lambda表达式
        dimension: 数据的维度，主要作用是为了防止程序因数据错误而崩溃
        
    def add(self, point):
        这个函数主要用于向已有的kdtree里添加一个节点
        在当前节点判断如果为空，则data赋值为point；否则，将point指定维度的坐标值与data相比，
        小于则往二叉树的左边走；大于则往二叉树的右边走；往左边走的时候如果是空结点，
        就调用subnode函数创建子结点，并且poin幅值给这个子节点.
        如果不是空结点，则将这个子结点视为当前结点。
        可以预料之后的行为与上述相同.然后不停的循环。
        
    def creat_subnode(self, data):
        在类的成员方法里创建自己的实例，用到了self.__class__()这个函数
        创建子结点，返回一个KDNode对象的实例

    def extreme_child(self, sel_func, axis):
        

    def find_replacement(self):
        
'''