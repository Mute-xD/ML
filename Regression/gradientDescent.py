import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv", delimiter=",")
xData = data[:, 0]
yData = data[:, 1]
plt.scatter(xData, yData)
plt.show()
lr = 0.0001  # 学习率
b = 0  # 初始截距
k = 0  # 初始斜率
epochs = 50  # 最大迭代次数


def cost(b_, k_, x_data, y_data):  # 代价函数
    diff_sum = 0
    for i in range(0, len(x_data)):
        diff_sum += (y_data[i] - (k_ * x_data[i] + b_)) ** 2  # (y - y估) ** 2

    return diff_sum / float(len(x_data)) / 2.0  # 1 / 2m


def runner(x_data, y_data, b_, k_, lr_, epochs_):
    mount = float(len(x_data))
    for i in range(epochs_):
        b_grad = 0
        k_grad = 0
        for j in range(0, len(x_data)):
            b_grad += (1 / mount) * (((k_ * x_data[j]) + b) - y_data[j])
            k_grad += (1 / mount) * (((k_ * x_data[j]) + b) - y_data[j]) * x_data[j]
        # 更新 b, k
        b_ = b_ - (lr_ * b_grad)
        k_ = k_ - (lr_ * k_grad)
        if i % 5 == 0:
            print("epochs", i)
            plt.plot(x_data, y_data, "b.")
            plt.plot(x_data, k_ * x_data + b_, "r")
            plt.show()
    return b_, k_


print("Starting, b = {0}, k = {1}, diff = {2}".format(b, k, cost(b, k, xData, yData)))
b, k = runner(xData, yData, b, k, lr, epochs)
print("Finished, gen = {0}, b = {1}, k = {2}, diff = {3}".format(epochs, b, k, cost(b, k, xData, yData)))
plt.plot(xData, yData, "b.")
plt.plot(xData, k * xData + b, "r")
plt.show()
