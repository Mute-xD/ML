"""

RandomWalk
    可以和元胞自动机联用

"""

# import numpy as np
from random import choice
import matplotlib.pyplot as plt


class RandomWalk:
    def __init__(self, point_count, start_point=None, distance_x=4, distance_y=None):
        if start_point is None:
            start_point = [0, 0]
        if distance_y is None:
            distance_y = distance_x
        self.distanceX = distance_x  # 一次最多走几步
        self.distanceY = distance_y
        self.pointCount = point_count
        self.listX = [start_point[0]]
        self.listY = [start_point[1]]

    def walk(self):
        while len(self.listX) < self.pointCount:
            direction_x = choice([1, -1])
            distance_x = choice(range(self.distanceX + 1))
            step_x = direction_x * distance_x

            direction_y = choice([1, -1])
            distance_y = choice(range(self.distanceY + 1))
            step_y = direction_y * distance_y

            if step_x is 0 and step_y is 0:
                continue
            current_x = self.listX[-1] + step_x
            current_y = self.listY[-1] + step_y
            self.listX.append(current_x)
            self.listY.append(current_y)

    def matplotlibPlot(self, isShow=False, isStartEndHighlight=False, *args, **kwargs, ):
        plt.scatter(self.listX, self.listY, *args, **kwargs)
        if isStartEndHighlight:
            plt.scatter(0, 0, c='g', edgecolors='none', s=100)
            plt.scatter(self.listX[-1], self.listY[-1], c='r', edgecolors='none', s=100)
        if isShow:
            plt.show()


if __name__ == '__main__':
    randomWalk1 = RandomWalk(50000, distance_x=5)
    randomWalk1.walk()
    randomWalk1.matplotlibPlot(False, True,
                               c=list(range(randomWalk1.pointCount)),
                               cmap='Blues', edgecolor='none', s=10)
    plt.show()
