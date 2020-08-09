"""

"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as prep
from sklearn.metrics import classification_report  # 评测指标

scale = True  # 标准化开关

data = np.genfromtxt('./LR-testSet.csv', delimiter=',')
xData = data[:, :-1]
yData = data[:, -1, np.newaxis]

if scale:
    xData = prep.scale(xData)


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


def sigmoid(x1):
    return 1.0 / (1 + np.exp(-x1))


def cost(x_mat, y_mat, weight):
    left = np.multiply(y_mat, np.log(sigmoid(x_mat * weight)))  # 此处multiply为点乘(按位相乘)
    right = np.multiply(1 - y_mat, np.log(1 - sigmoid(x_mat * weight)))
    return np.sum(left + right) / -(len(x_mat))


def gradAscend(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    print(x_mat)
    lr = 0.001
    epochs = 10000
    cost_list = []

    m, n = np.shape(x_mat)  # 行(m) 数据个数，列(n)权值个数
    weight = np.mat(np.ones((n, 1)))

    for i in range(epochs + 1):  # 0~9999 + 1
        h = sigmoid(x_mat * weight)
        weight_grad = x_mat.T * (h - y_mat) / m
        weight = weight - lr * weight_grad

        if i % 50 == 0:
            cost_list.append(cost(x_mat, y_mat, weight))
    return weight, cost_list


ws, costLists = gradAscend(xData, yData)
print(ws)


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
    x_mat = np.mat(x_data)
    weight = np.mat(weight)
    # return [1 if x >= 0.5 else 0 for x in sigmoid(x_mat * weight)]  # 和下面一个意思
    returning = []
    for x1 in sigmoid(x_mat * weight):
        if x1 >= 0.5:
            returning.append(1)
        else:
            returning.append(0)
    return returning


prediction = predict(xData, ws)
print(classification_report(yData, prediction))
