import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


class BPNeuralNetworkOCR:
    def __init__(self, xData, yData, layer=(64, 100, 10), learning_rate=0.2, epochs=10000, autoRun=True):
        self.xData, self.yData = xData, yData
        self.xTest, self.yTest = None, None
        self.xTrain, self.yTrain = None, None
        self.V, self.W = None, None
        self.layer = layer
        self.currentEpochs = 0
        self.epochs = epochs
        self.accuracyList = []

        self.LEARNING_RATE = learning_rate
        if autoRun:
            self.functionSequence()

    def labelBinaries(self):
        # 神经网络需要二值化的target数据
        self.yTrain = LabelBinarizer().fit_transform(self.yTrain)

    def dataSeparation(self):
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.xData, self.yData)

    def createWeight(self):
        self.V = np.random.random((self.layer[0] + 1, self.layer[1] + 1)) * 2 - 1
        self.W = np.random.random((self.layer[1] + 1, self.layer[2])) * 2 - 1  # 这里就没有偏置了

    def createBias(self):
        # 在xData加一列偏置，合并一列总有问题，这是个好办法
        temp_array = np.ones((self.xTrain.shape[0], self.xTrain.shape[1] + 1))
        temp_array[:, 0:-1] = self.xTrain
        self.xTrain = temp_array

    def update(self):
        i = np.random.randint(self.xTrain.shape[0])  # 随机选一个
        x = np.atleast_2d([self.xTrain[i]])

        l1 = self.sigmoid(np.dot(x, self.V))
        l2 = self.sigmoid(np.dot(l1, self.W))

        sigma_l2 = (self.yTrain[i] - l2) * self.dSigmoid(l2)
        sigma_l1 = np.dot(sigma_l2, self.W.T) * self.dSigmoid(l1)

        delta_w = self.LEARNING_RATE * np.dot(l1.T, sigma_l2)
        delta_v = self.LEARNING_RATE * np.dot(x.T, sigma_l1)

        self.W += delta_w
        self.V += delta_v

    def response(self):
        if self.currentEpochs % 100 == 0:
            prediction_list = []
            for i in range(self.xTest.shape[0]):
                prediction = self.predict(self.xTest[i])
                prediction_list.append(np.argmax(prediction))
            accuracy = np.mean(np.equal(prediction_list, self.yTest))
            print('Current Epochs:', self.currentEpochs, '    Accuracy:', accuracy)

    def predict(self, x):
        temp_array = np.ones(x.shape[0] + 1)
        temp_array[0:-1] = x
        x = np.atleast_2d(temp_array)

        l1 = self.sigmoid(np.dot(x, self.V))
        l2 = self.sigmoid(np.dot(l1, self.W))

        return l2

    def runner(self):
        for _ in range(self.epochs + 1):
            self.currentEpochs = _
            self.update()
            self.response()

    def functionSequence(self):
        self.dataSeparation()
        self.labelBinaries()
        self.createWeight()
        self.createBias()
        self.runner()

    @staticmethod
    def sigmoid(_x):
        return 1 / (1 + np.exp(-_x))

    @staticmethod
    def dSigmoid(_x):
        return _x * (1 - _x)


digits = load_digits()

data, target = digits.data, digits.target
# 归一化    因为要用sigmoid，xData太大，都趋近于一
data -= data.min()
data /= data.max()
bp_nn_ocr = BPNeuralNetworkOCR(data, target)
