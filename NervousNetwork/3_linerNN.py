"""
线性神经网络
"""
import singleLayerPerceptron
import numpy as np


class LinerNN(singleLayerPerceptron.SingleLayerPerceptron):
    pass

    def update(self):
        self.output = self.x.dot(self.W)  # 激活函数改变为 purelin 即 y = x
        delta_w = self.learningRate * self.x.T.dot(self.y - self.output) / int(self.x.shape[0])
        print('Current Delta_W: ', delta_w.squeeze())
        self.W += delta_w

    def run(self):
        for _ in range(20):
            self.update()
            self.plot()


if __name__ == '__main__':
    x1 = np.array([[1, 3, 3],
                   [1, 4, 3],
                   [1, 1, 1],
                   [1, 0, 2]])
    y1 = np.array([[1], [1], [-1], [-1]])
    liner_nn = LinerNN(x1, y1)
