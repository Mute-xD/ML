"""
标准方程法实现
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv", delimiter=",")
xData = np.atleast_2d(data[:, 0]).T
yData = np.atleast_2d(data[:, 1]).T
# plt.scatter(xData, yData)
# plt.show()


xData1 = np.concatenate((np.ones((100, 1)), xData), axis=1)  # 和全一数组合并
print(xData1.shape)


def weight(x, y):
    x_mat = np.mat(x)
    y_mat = np.mat(y)
    x_mat2 = x_mat.T * x_mat  # 平方
    if np.linalg.det(x_mat2) == 0:  # 矩阵的行列式求值
        raise Exception("Can`t do inverse")  # 异常
    return x_mat2.I * x_mat.T * y_mat  # .I 求逆矩阵


ws = weight(xData1, yData)
print(ws)

xTest = np.array([[20], [80]])
yTest = ws[0] + xTest * ws[1]
plt.plot(xData, yData, "b.")
plt.plot(xTest, yTest, "r")
plt.show()
