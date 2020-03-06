"""

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report  # 评测指标
import sklearn.preprocessing as prep

scale = False  # 标准化开关  (True 有bug，但我)

data = np.genfromtxt('./LR-testSet.csv', delimiter=',')
xData = data[:, :-1]
yData = data[:, -1, np.newaxis]


def plot():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    # init
    for i in range(len(xData)):
        if yData[i] == 0:
            x0.append(xData[i, 1])
            y0.append(xData[i, 2])
        else:
            x1.append(xData[i, 1])
            y1.append(xData[i, 2])
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')


xData = np.concatenate((np.ones((100, 1)), xData), axis=1)
print(xData.shape)
print(yData.shape)


def sigmoid(x1):
    return 1.0 / (1 + np.exp(-x1))


def cost(xMat, yMat, weight):
    lSide = np.multiply(yMat, np.log(sigmoid(xMat * weight)))  # 此处multiply为点乘(按位相乘)
    rSide = np.multiply(1 - yMat, np.log(1 - sigmoid(xMat * weight)))
    return np.sum(lSide + rSide) / -(len(xMat))


def gradAscend(xArr, yArr):
    if scale:
        xArr = prep.scale(xArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    lr = 0.001
    epochs = 10000
    costList = []

    m, n = np.shape(xMat)  # 行(m) 数据个数，列(n)权值个数
    weight = np.mat(np.ones((n, 1)))

    for i in range(epochs + 1):  # 0~9999 + 1
        h = sigmoid(xMat * weight)
        weightGrad = xMat.T * (h - yMat) / m
        weight = weight - lr * weightGrad

        if i % 50 == 0:
            costList.append(cost(xMat, yMat, weight))
    return weight, costList


ws, costLists = gradAscend(xData, yData)
print(ws)

if not scale:
    plot()
    xTest = [[-4], [3]]
    yTest = (-ws[0] - xTest * ws[1]) / ws[2]  # w0 + x1w1 +x2w2 = 0
    plt.plot(xTest, yTest, 'k')
    plt.show()


x = np.linspace(0, 10000, 201)
plt.plot(x, costLists, 'r')
plt.title("Train")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()


def predict(x_data, weight):
    if scale:
        x_data = prep.scale(x_data)
    xMat = np.mat(xData)
    weight = np.mat(weight)
    # return [1 if x >= 0.5 else 0 for x in sigmoid(xMat * weight)]  # 和下面一个意思
    returning = []
    for x1 in sigmoid(xMat * weight):
        if x1 >= 0.5:
            returning.append(1)
        else:
            returning.append(0)
    return returning


prediction = predict(xData, ws)
print(classification_report(yData, prediction))
