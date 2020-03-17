import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import linear_model

scale = False  # 标准化开关

data = np.genfromtxt("./LR-testSet.csv", delimiter=",")
xData = data[:, :-1]
yData = data[:, -1]


def plot():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for i in range(len(xData)):
        if yData[i] == 0:
            x0.append(xData[i, 0])
            y0.append(xData[i, 1])
        else:
            x1.append(xData[i, 0])
            y1.append(xData[i, 1])

    scatter0 = plt.scatter(x0, y0, c="b", marker="o")
    scatter1 = plt.scatter(x1, y1, c="r", marker="x")

    plt.legend(handles = [scatter0, scatter1], labels=[""])

