"""

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report  # 评测指标
import sklearn.preprocessing  # 标准化

scale = False  # 标准化开关

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
            x0.append(xData[i, 0])
            y0.append(xData[i, 1])
        else:
            x1.append(xData[i, 0])
            y1.append(xData[i, 1])
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')
    plt.show()


# plot()
xData = np.concatenate((np.ones((100, 1)), xData), axis=1)
print(xData.shape)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


