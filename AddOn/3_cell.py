"""

元胞自动机

"""

import numpy as np
import matplotlib.pyplot as plt


class Cell:
    def __init__(self):
        self.space = np.zeros((100, 100))
        self.spaceShape = self.space.shape
        self.space[0][0] = 1

        self.UPDATE_LIMIT = 50
        self.POLLUTION_RATE = 0.3
        self.RECOVER_RATE = 0.2
        self.RESPONSE_IN_EACH = 10
        """
        UPDATE_LIMIT = 迭代次数
        POLLUTION_RATE = 污染系数
        RECOVER_RATE = 恢复系数
        RESPONSE_IN_EACH = 每N次迭代Plot一次
        """

    def update(self):
        for epochs in range(self.UPDATE_LIMIT):
            for y in range(self.spaceShape[1]):
                for x in range(self.spaceShape[0]):
                    if np.any(self.findNearBy(x, y) != self.space[x][y]):
                        if np.random.rand() < self.POLLUTION_RATE:
                            self.space[x][y] = 1
                    if np.random.rand() < self.RECOVER_RATE:
                        self.space[x][y] = 0
            if (epochs+1) % self.RESPONSE_IN_EACH == 0:
                self.ui(epochs+1)

    def findNearBy(self, x, y):
        nearby_list = [self.getCells(x + delta_x, y + delta_y)
                       for delta_x in [-1, 0, 1] for delta_y in [-1, 0, 1] if not (delta_x is 0 and delta_y is 0)]
        return [i for i in nearby_list if i is not None]

    def getCells(self, x, y):
        if x < 0 or x > self.spaceShape[0] - 1 or y < 0 or y > self.spaceShape[1] - 1:
            return None
        else:
            return self.space[x][y]

    def ui(self, epochs):

        title = 'Current Epochs: ' + str(epochs)
        plt.title(title)
        plt.imshow(self.space, cmap='Set3')
        plt.show()


if __name__ == '__main__':
    cell = Cell()
    cell.update()
