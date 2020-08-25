"""

求函数的积分
    http://liao.cpython.org/scipy18/

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.integrate


# 已知函数型求积分


def func1(x):
    return x + 1


res = scipy.integrate.quad(func1, 1, 2)  # 上下限1-2 返回值(value, error)
print(res)


#################################################################################

def func2(x, a, b):
    return a * x + b


res = scipy.integrate.quad(func2, 1, 2, args=(1, 1))  # 通过arg为函数传参
print(res)


#########################################################################################
# 积分函数有断点

def func3(x):
    return 1 / np.sqrt(abs(x))


# res = scipy.integrate.quad(func3, -1, 1)  # 炸了，0是断点
# print(res)
res = scipy.integrate.quad(func3, -1, 1, points=[0])
print(res)
x_ = np.arange(-1, 1, 0.002)
plt.plot(x_, func3(x_))
plt.fill_between(x_, func3(x_), alpha=0.5)
plt.ylim(0, 25)


# plt.show()


##############################################################################
# 假设原函数未知

def func4(x):
    return np.sqrt(x)


x_ = np.arange(0, 2, 0.02)  # 0-2 的多个数据点
y_ = func4(x_)
value = scipy.integrate.trapz(y_, x_)
print(value)


#####################################################################################
# 多重积分
# SciPy下的二重积分可以用dblquad函数来计算、三重积分可以用tplquad函数来计算
# 多重积分可以使用nquad函数
def func5(x, y):  # 二重积分 原方程在这打不出来
    return x * y


res = scipy.integrate.dblquad(func5, 1, 2, lambda x: 1, lambda x: x)
print(res)


def func6(x, y, z):  # 三重积分 原方程在这打不出来
    return x


res = scipy.integrate.tplquad(func6, 0, 1,
                              lambda x: 0,
                              lambda x: (1 - x) / 2,
                              lambda x, y: 0,
                              lambda x, y: 1 - x - 2 * y)
print(res)
