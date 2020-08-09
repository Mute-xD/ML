"""
岭回归实现
标准方程法
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("longley.csv", delimiter=",")

xData = data[1:, 2:]
yData = np.atleast_2d(data[1:, 1]).T
print(xData.shape)
print(yData.shape)
xData = np.concatenate((np.ones((16, 1)), xData), axis=1)
print(xData.shape)


def weights(x_arr, y_arr, ridge_arg=0.2):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    x_mat2 = x_mat.T * x_mat
    rx = x_mat2 + np.identity(x_mat.shape[1]) * ridge_arg
    if np.linalg.det(rx) == 0:
        raise Exception("Can`t do inverse")
    ws = rx.I * x_mat.T * y_mat
    return ws


ws = weights(xData, yData)
print(ws.shape)
predict = np.mat(xData) * np. mat(ws)
print(predict)
plt.plot(xData[:, -2], yData, "b.")
plt.plot(xData[:, -2], predict, "r")
plt.show()
