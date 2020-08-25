import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

"""
最小二乘拟合
fitting()
    level = 拟合函数的次数
"""


def targetFunc(x_):  # 这是目标函数
    return 2 * np.sin(2 * np.pi * x_)


def residuals(para, x_, y_):
    fun = np.poly1d(para)
    return y_ - fun(x_)


def fitting(level, x_data, y_data):
    pars = np.random.rand(level + 1)
    return optimize.leastsq(residuals, pars, args=(x_data, y_data))


xData = np.linspace(0, 1, 10)
yData = [np.random.normal(0, 0.1) + i for i in targetFunc(xData)]
xPlot = np.linspace(0, 1, 100)
yPlot = targetFunc(xPlot)

response = fitting(3, xData, yData)
if response[1] in [1, 2, 3, 4]:
    print('OK')
    response = response[0]
else:
    raise Exception('Fitting Failed')
plt.plot(xPlot, yPlot, label='real line')
plt.scatter(xData, yData, label='real points')
plt.plot(xPlot, np.poly1d(response)(xPlot), label='fitting line')
plt.legend()
plt.show()
print(np.poly1d(response))
