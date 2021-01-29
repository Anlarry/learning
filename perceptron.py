import numpy as np

class A:
    def __init__(self, a):
        self.a = a

class Perceptron:
    def __init__(self, eta:float, times:int):
        self.eta = eta
        self.times = times
        self._wb = []
    def Fit(self, X:np.ndarray, Y:np.ndarray)  :
        self.X = X
        self.Y = Y
        w = np.zeros(X[0].shape)
        b = 1
        self._wb.append((w,b))
        for i in range(self.times) :
            if (pt := self.ErrPt(w, b, X, Y)) != None:
                w = w + self.eta * pt[0] * pt[1]
                b = b + self.eta * pt[1]
                self._wb.append((w, b))
            else :
                break
    def Predict(self, x) :
        w, b = self._wb[-1]
        return w.dot(x) + b
    def ErrPt(self, w, b, X, Y):
        for x, y in zip(X, Y):
            if (w.dot(x) + b) * y < 0 :
                return x, y
        return None