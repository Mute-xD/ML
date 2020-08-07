import numpy as np
import matplotlib.pyplot as plt


class BPNeuralNetwork:
    def __init__(self, xData, yData):
        self.x = xData
        self.y = yData
        self.o = None
        self.W = None
        self.V = None

        self.LEARNING_RATE = 0.12
        self.HIDDEN_LAYER = 4
        self.EXPECTED_EPOCHS = 20000

        self.currentEpochs = 0
        self.errorList = []
        self.functionSequence()

    def functionSequence(self):
        self.createWeight()
        self.run()
        self.plot()

    def createWeight(self):
        self.V = (np.random.random((self.x.shape[1], self.HIDDEN_LAYER)) - 0.5) * 2
        self.W = (np.random.random((self.HIDDEN_LAYER, 1)) - 0.5) * 2

    @staticmethod
    def sigmoid(x_):
        return 1 / (1 + np.exp(-x_))

    @staticmethod
    def dSigmoid(x_):
        return x_ * (1 - x_)

    def update(self):
        l1 = self.sigmoid(np.dot(self.x, self.V))  # hidden layer 输出
        l2 = self.sigmoid(np.dot(l1, self.W))  # output layer 输出

        sigma_l2 = (self.y.T - l2) * self.dSigmoid(l2)
        sigma_l1 = np.dot(sigma_l2, self.W.T) * self.dSigmoid(l1)

        delta_w = self.LEARNING_RATE * l1.T.dot(sigma_l2)
        delta_v = self.LEARNING_RATE * self.x.T.dot(sigma_l1)

        self.W += delta_w
        self.V += delta_v
        self.errorList.append(np.mean(np.abs(self.y.T - l2)))
        if self.currentEpochs % 500 == 0:
            print('In Epochs:', self.currentEpochs + 1)
            print('Current DeltaW: \n', delta_w)
            print('Current DeltaV: \n', delta_v)
            print('Error: ', np.mean(np.abs(self.y.T - l2)))
            print('--------------------------------------------------------------------------------------------')
        if self.currentEpochs == self.EXPECTED_EPOCHS - 1:
            print('Current L2: ', l2)

    def run(self):
        for self.currentEpochs in range(self.EXPECTED_EPOCHS):
            self.update()

    def plot(self):
        plt.plot(self.errorList)
        plt.show()


x1 = np.array([[1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])
y1 = np.array([[0, 1, 1, 0]])  # sigmoid 到不了y负半轴，这里标记得改
bp_nn = BPNeuralNetwork(x1, y1)

