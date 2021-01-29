from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import *
from PySide6.QtCore import *
import sys

class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
    def InCircle(self, x, y):
        return (x-self.x)**2 + (y-self.y)**2 < self.r**2

class Widgets(QtWidgets.QWidget):
    def __init__(self, circle):
        super().__init__()
        self.circle = circle
        self.setFixedWidth(1000)
        self.setFixedHeight(1000)
    def paintEvent(self, paintEvent):
        painter = QPainter(self)
        for c in self.circle:
            painter.drawEllipse(c.x-c.r, c.y-c.r, c.r*2, c.r*2 )
        painter.setPen(QColor(255, 0, 0))
        for i in range(self.width()):
            for j in range(self.height()):
                if sum(c.InCircle(i, j) for c in self.circle) >= 2:
                    painter.drawPoint(i, j)
                    # print(i, j)
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    circles = [Circle(*x) for x in
        [(500, 635.11, 300), (338.19, 517.56, 300), (400, 327.35, 300),(600, 327.35, 300), (661.80, 517.56, 300)]
    ]
    widgt = Widgets(circles)
    widgt.show()
    sys.exit(app.exec_())