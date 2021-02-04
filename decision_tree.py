import numpy as np
from collections import Counter
import math
from typing import List, Tuple
import abc

modes = {}

def register(cls_:object):
    modes[cls_.__name__] = cls_
    return cls_

def DecisionTree(X, Y, epsilon, mode='C45'):
    return modes[mode](X, Y, epsilon)

class Node:
    def __init__(self, feature,child:dict, label=None):
        self.feature = feature
        self.child = child
        self.label = label
    def __getitem__(self, item):
        return self.child[item]

class DecisionTreeBase:
    Data = List[Tuple[np.ndarray, ]]
    @classmethod
    def Entropy(cls, data:Data):
        cnt = Counter([x[1] for x in data])
        res = 0
        for item in cnt.items():
            res += -item[1]/len(data) * math.log2(item[1] / len(data))
        return res
    @classmethod
    def MutualInfo(cls, data:Data, feature):
        child = {}
        e0 = DecisionTreeBase.Entropy(data)
        for each in data:
            if (key := each[0][feature]) in child:
                child[key].append(each)
            else :
                child[key] = [each]
        e1 = 0
        for item in child.items():
            e1 += -len(item[1]) / len(data) * DecisionTreeBase.Entropy(item[1])
        return e0 - e1
    def __init__(self, X, Y, epsilon):
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.root = self._Fit(list(zip(X, Y)))
    def _Fit(self, data:Data) -> Node:
        feature = self._BestFeature(data)
        if not feature:
            return Node(feature, None, Counter([x[1] for x in data]).most_common(1)[0][0])
        child_data = {}
        for each in data:
            if (key := each[0][feature]) in child_data:
                child_data[key].append(each)
            else:
                child_data[key] = [each]
        child = {}
        for item in child_data.items():
            child[item[0]] = self._Fit(item[1])
        return Node(feature, child)
    @abc.abstractmethod
    def _BestFeature(self, data:Data):
        pass
    def Predict(self, x):
        node = self.root
        while node.child:
            node = node.child[x[node.feature]]
        return node.label

@register
class ID3(DecisionTreeBase):
    def _BestFeature(self, data:DecisionTreeBase.Data):
        feature = None
        gain = -1
        for i in range(len(data[0][0])):
            if (val := self.MutualInfo(data, i)) > max(gain, self.epsilon):
                gain = val
                feature = i
        return feature

@register
class C45(DecisionTreeBase):
    @classmethod
    def FeatureEntropy(cls, data:DecisionTreeBase.Data, feature):
        cnt = Counter([x[0][feature] for x in data])
        res = 0
        for each in cnt.items():
            res += -each[1] / len(data) * math.log2(each[1] / len(data))
        return res
    def _BestFeature(self, data:DecisionTreeBase.Data):
        e0 = DecisionTreeBase.Entropy(data)
        feature = None
        gain = -1
        for i in range(len(data[0][0])):
            try:
                if (val := self.MutualInfo(data, i) / self.FeatureEntropy(data, i)) \
                      > max(gain, self.epsilon):
                    gain = val
                    feature = i
            except ZeroDivisionError:
                pass
        return feature