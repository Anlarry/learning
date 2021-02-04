import numpy as np
import heapq
import copy
import re
from collections import Counter

class KHeap(list):
    """
    Big root heap, maintain k smallest element
    """
    def __init__(self, k):
        super().__init__([])
        assert k != 0
        self.k = k
    def CheckPush(self, x):
        # print("x", x)
        if self.__len__() < self.k:
            self._Push(x)
        elif self[0] < -x:
            self.Pop()
            self._Push(x)
    def RootVal(self):
        return -self[0]
    def _Push(self, x):
        heapq.heappush(self, -x)
    def Pop(self):
        return -heapq.heappop(self)

class Node:
    def __init__(self, data:np.ndarray, d, l = None, r = None, p = None):
        self.l = l
        self.r = r
        self.p = p
        self.d = d
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __str__(self):
        return self.data.__str__()

def DisNodeFactory(target:np.ndarray):
    class DisNode:
        def __init__(self, node:Node):
            self.dis = np.linalg.norm(node.data-target, ord=2)
            self.node = node
        def __neg__(self):
            res = copy.deepcopy(self)
            res.dis = -res.dis
            return res
        def __lt__(self, other):
            return self.dis < other.dis
        def __getattr__(self, item):
            if re.match(r'^__.+__$', item):
                raise AttributeError
            return getattr(self.node, item)
        def __str__(self):
            return self.node.data.__str__() + " -> dis {:.2f}".format(self.dis)
    return DisNode

class KDTree:
    def __init__(self, X):
        self.k = len(X[0])
        self.root = self._Build(X, 0)
    def _Build(self, X:list[np.ndarray], d:int, p:Node=None) -> Node :
        size = len(X)
        if size <= 0 :
            return None
        X = sorted(X, key=lambda x : x[d])
        res = Node(X[size//2], d, p = p)
        res.l = self._Build(X[:size // 2], (d + 1) % self.k, res)
        res.r = self._Build(X[size//2 + 1:], (d + 1) % self.k, res)
        return res
    def Nearest(self, x:np.ndarray, k = 1) -> KHeap:
        nearest = self._search(x, self.root)
        que = [(self.root, nearest)]
        heap = KHeap(k)
        DisNode = DisNodeFactory(x)
        heap.CheckPush(DisNode(nearest))
        while len(que) != 0:
            root, cur_node = que[0]
            que.pop(0)
            while cur_node != root:
                # print("traverse {}, {}".format(cur_node, root))
                heap.CheckPush(DisNode(cur_node))
                brother = cur_node.p.l if cur_node is cur_node.p.r else cur_node.p.r
                d = cur_node.p.d
                # if brother and abs(cur_node.p[d] - x[d]) < np.linalg.norm(x-nearest.data, ord=2):
                # print("cmp", abs(cur_node.p[d] - x[d]), heap.RootVal())
                if brother and abs(cur_node.p[d] - x[d]) < heap.RootVal().dis:
                    nearest_ = self._search(x, brother)
                    que.append((brother, nearest_))
                cur_node = cur_node.p
            heap.CheckPush(DisNode(cur_node))
        return heap
    def _search(self, x:np.ndarray, root:Node) -> Node:
        d = root.d
        cur_node = root
        while cur_node.l or cur_node.r:
            if not cur_node.l:
                cur_node = cur_node.r
                break
            if not cur_node.r:
                cur_node = cur_node.l
                break
            if cur_node[d] < x[d]:
                cur_node = cur_node.r
            else:
                cur_node = cur_node.l
            d = (d + 1) % self.k
        return cur_node

def LabelArrayFactory(x:np.ndarray, y_):
    class LabelArray(np.ndarray):
        y = y_
    return LabelArray(x.shape, x.dtype, x)

class KNN:
    def __init__(self, X, Y, k):
        X = [LabelArrayFactory(x, y) for x, y in zip(X, Y)]
        self.kdtree = KDTree(X)
        self.k = k
    def Predict(self, x):
        self.nearest = self.kdtree.Nearest(x, self.k)
        for node in self.nearest:
            node.dis = abs(node.dis)
        lables = Counter([node.data.y for node in self.nearest])
        best = sorted(lables.items(), key=lambda x:x[1], reverse=True)[0][0]
        return best
