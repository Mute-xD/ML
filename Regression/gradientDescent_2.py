import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt("Delivery.csv", delimiter=",")
print(data)
xData = data[:, 0:2]
yData = data[:, 2]
print(xData)
print(yData)

lr = 0.0001

arg0 = 0
arg1 = 0
arg2 = 0
# y = arg0 + arg1 * x0 +arg2 * x1
epochs = 1000


def diff(x_data, y_data, arg0_, arg1_, arg2_):
    diff_sum = 0
    for i in range(0, len(x_data)):
        diff_sum += (y_data[i] - (arg1_ * x_data[i, 0] + arg2_ * x_data[i, 1] + arg0_)) ** 2
    return diff_sum / float(len(x_data))


def runner(x_data, y_data, arg0_, arg1_, arg2_, lr_, epochs_):
    m = float(len(x_data))
    for i in range(epochs_):
        arg0_grad = 0
        arg1_grad = 0
        arg2_grad = 0
        for j in range(0, len(x_data)):
            arg0_grad += (1 / m) * ((arg0_ + arg1_ * x_data[j, 0] + arg2_ * x_data[j, 1]) - y_data[j])
            arg1_grad += (1 / m) * ((arg0_ + arg1_ * x_data[j, 0] + arg2_ * x_data[j, 1]) - y_data[j]) * x_data[j, 0]
            arg2_grad += (1 / m) * ((arg0_ + arg1_ * x_data[j, 0] + arg2_ * x_data[j, 1]) - y_data[j]) * x_data[j, 1]
        arg0_ = arg0_ - (lr_ * arg0_grad)
        arg1_ = arg1_ - (lr_ * arg1_grad)
        arg2_ = arg2_ - (lr_ * arg2_grad)
    return arg0_, arg1_, arg2_


print("Starting arg0 = {0} arg1 = {1} arg2 = {2} diff = {3}"
      .format(arg0, arg1, arg2, diff(xData, yData, arg0, arg1, arg2)))
arg0, arg1, arg2 = runner(xData, yData, arg0, arg1, arg2, lr, epochs)
print("Finished y = {0} + {1} x0 + {2} x1 diff = {3}"
      .format(arg0, arg1, arg2, diff(xData, yData, arg0, arg1, arg2)))

ax = plt.figure().add_subplot(111, projection="3d")
ax.scatter(xData[:, 0], xData[:, 1], yData, c="r", marker="o", s=100)
x0 = xData[:, 0]
x1 = xData[:, 1]
x0, x1 = np.meshgrid(x0, x1)
z = arg0 + x0 * arg1 + x1 * arg2

ax.plot_surface(x0, x1, z)
ax.set_xlabel("miles")
ax.set_ylabel("deliveries")
ax.set_zlabel("time")
plt.show()
