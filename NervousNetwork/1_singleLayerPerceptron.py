"""
异或 是非线性的，无法用单层神经网络解决
"""
import numpy as np
import matplotlib.pyplot as plt


class SingleLayerPerceptron:
    def __init__(self, xData, yData):
        self.x = xData
        self.y = yData
        self.output = None
        self.W = None
        self.learningRate = 0.11
        self.functionSequence()

    def functionSequence(self):
        self.createW()
        self.run()

    def createW(self):
        self.W = (np.random.random([self.x.shape[1], self.y.shape[1]]) - 0.5) * 2  # 处理到 -1 ~ 1 之间

    def update(self):
        self.output = np.sign(self.x.dot(self.W))
        delta_w = self.learningRate * self.x.T.dot(self.y - self.output) / int(self.x.shape[0])
        print('Current Delta_W: ', delta_w.squeeze())
        self.W += delta_w

    def run(self):
        for i in range(100):
            self.update()
            if (self.output == self.y).all():
                print('OK')
                print('done in', i, 'round(s)')
                print('Now W is: ', self.W.squeeze())
                break

    def plot(self):
        line_x = [min(self.x[:, 1]) - 1, max(self.x[:, 1]) + 1]
        k = - self.W[1] / self.W[2]  # ax + by + c = 0
        b = - self.W[0] / self.W[2]
        plt.plot(line_x, k * line_x + b, c='r')
        plt.scatter(self.x[:, 1], self.x[:, 2])
        plt.show()


if __name__ == '__main__':
    x1 = np.array([[1, 3, 3],
                  [1, 4, 3],
                  [1, 1, 1],
                  [1, 0, 2]])
    y1 = np.array([[1], [1], [-1], [-1]])
    slp = SingleLayerPerceptron(x1, y1)
    # slp.plot()
    print(x1[0].shape)
    x1[0].reshape(6)
    print(x1[0].shape)
