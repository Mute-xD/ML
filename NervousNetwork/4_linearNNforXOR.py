import linerNN
import numpy as np
import matplotlib.pyplot as plt


class LinerNNXor(linerNN.LinerNN):
    def __init__(self, xData, yData):
        super(LinerNNXor, self).__init__(xData, yData)

    def functionSequence(self):
        self.createNonLinear()
        self.createW()
        self.run()

    def createNonLinear(self):
        temp_array = np.zeros((self.x.shape[0], 6))
        for i in range(self.x.shape[0]):
            temp_array[i] = np.append(self.x[i],
                                      np.array((self.x[i][1] ** 2, self.x[i][1] * self.x[i][2], self.x[i][2] ** 2)))
        # w0 + x1w1 + x2w2 + x1^2*w3 + x1x2*w4 + x2^2*w5 = 0
        self.x = temp_array

    def plot(self):
        line_space = np.linspace(min(self.x[:, 1]) - 1, max(self.x[:, 1]) + 1)
        line_space_y1 = []
        line_space_y2 = []
        # 梦回高中  求根公式
        for x in line_space:
            a = self.W[5]
            b = self.W[2] + x * self.W[4]
            c = self.W[0] + x * self.W[1] + x*x * self.W[3]
            y1_ = (-b + np.sqrt(b*b - 4*a*c)) / (2*a)
            y2_ = (-b - np.sqrt(b*b - 4*a*c)) / (2*a)
            line_space_y1.append(y1_)
            line_space_y2.append(y2_)
        plt.plot(line_space, line_space_y1, 'r')
        plt.plot(line_space, line_space_y2, 'r')
        plt.scatter(self.x[:, 1], self.x[:, 2])
        plt.show()

    def run(self):  # 不验证收敛
        for _ in range(1000):
            self.update()
        self.plot()


x1 = np.array([[1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])
y1 = np.array([[1], [-1], [-1], [1]])
liner_nn_xor = LinerNNXor(x1, y1)
