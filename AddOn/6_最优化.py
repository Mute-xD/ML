import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def rosen(x):  # 二维的 Rosenbrock 函数
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] * x[0]) ** 2


def plot():
    xx, yy = np.meshgrid(np.arange(-2, 2, 0.05),
                         np.arange(-2, 2, 0.05))
    zz = np.array([rosen(np.c_[xx.flatten(), yy.flatten()][i]) for i in range(xx.flatten().shape[0])]).reshape(xx.shape)
    axes = plt.figure().add_subplot(projection='3d')
    axes.plot_surface(xx, yy, zz)
    plt.show()


def jacobi(x):  # 雅可比矩阵（偏导）
    result = np.zeros(2)
    result[0] = 2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1)
    result[1] = 200 * (x[1] - x[0] ** 2)
    return result


def hessian(x):  # Hij = (∂^2 f) / (∂xi * ∂xj)  黑塞矩阵
    pass


x0 = np.random.rand(2) * 2
"""
https://blog.csdn.net/jiang425776024/article/details/87885969
https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html#nelder-mead-simplex-algorithm-method-nelder-mead
http://liao.cpython.org/scipytutorial05/
minimize的各种method
无约束
    Nelder-Mead
        method='Nelder-Mead'
        单纯形
        Nelder-Mead算法主要应用于求解一些非线性、导函数未知的最大值或最小值问题
        若函数有n个变量，则数据集合（simplex）需要构建n+1个元素。利用这n+1个元素，不停地替换掉函数值最大（小）的元素，同时维护更新中心点的值，
        当最终的函数值满足容忍条件时即可得出近似解的结果
    BFGS
        method='BFGS'
        一种拟牛顿法，指用BFGS矩阵作为拟牛顿法中的对称正定迭代矩阵的方法
        由于BFGS法对一维搜索的精度要求不高，并且由迭代产生的BFGS矩阵不易变为奇异矩阵，因而BFGS法比DFP法在计算中具有更好的数值稳定性
    Newton-Conjugate Gradient
        method='Newton-CG'
        “共轭梯度法”是一种特殊的“共轭方向法”。既然叫共轭梯度法，它与梯度必然是有关系的。
        共轭方向法与梯度也有关系——共轭方向法利用了目标函数的梯度信息（梯度与方向的积满足“下降”条件）。共轭梯度法与此关系有所区别：
        用当前点的负梯度方向，与前面的搜索方向进行共轭化，以得到新的搜索方向
约束
    SLSQP
        method='method='SLSQP'
        Sequential Least SQuares Programming
        
    
"""
# plot()
print('单纯形法\n')
print(optimize.minimize(rosen, x0, method='Nelder-Mead'))

print('\n拟牛顿法：BFGS\n')
print(optimize.minimize(rosen, x0, method='BFGS', jac=jacobi))

print('\n牛顿法：Newton-CG\n')
print(optimize.minimize(rosen, x0, method='Newton-CG', jac=jacobi))  # 还得传一个hessian 矩阵，但我没看明白


def func(x):  # 2x^2 + 3y^2
    return 2 * x[0] ** 2 + 3 * x[1] ** 2


def dFunc(x):
    return [4 * x[0], 6 * x[1]]


con1 = {'type': 'eq',  # x-y + 1= 0
        'fun': lambda x: x[0] - x[1] + 1}
con2 = {'type': 'ineq',
        'fun': lambda x: x[1] - 2}  # y -2 >= 0
cons = [con1, con2]

print('\nSLSQP 动态规划\n')
print(optimize.minimize(func, np.array([-1.0, 2.0]), jac=dFunc, constraints=cons, method='SLSQP'))

print('\n单变量函数最小化 暴力\n')
print(optimize.minimize_scalar(lambda x: (x - 2) * (x + 1) ** 2, method='brent'))


def func1(x):
    return x ** 2 + 30 * np.sin(x)


print('\n局部极小值, -5, 0\n')
print(optimize.minimize_scalar(func1, method='bounded', bounds=(-5, 0)))

print('\n局部极小值, 0, 5\n')
print(optimize.minimize_scalar(func1, method='bounded', bounds=(0, 5)))

print('\n局部极小值, -5, 5\n')
print(optimize.minimize_scalar(func1, method='bounded', bounds=(-5, 5)))

print('\n求根\n')
print(optimize.root(func1, np.array([0, -3.5, 3, 5])))
# method='Krylov'适合变量较多的方程(组)的求根，速度要比method='hybr'(root函数默认求根方式)快，
# 原因是root函数求根时设置method='Krylov'，其内部使用了雅克比迭代法求根


def plot1():  # numpy 的一套数据类型转的真乱套，三个函数，三种get x的方法，令人折服
    x_ = np.arange(-10, 10, 0.05)
    min_global = optimize.brute(func1, ([-10, 10, 0.1], ))
    min_local = optimize.fminbound(func1, 0, 10)
    root = optimize.root(func1, np.array([0, -3.5, 3, 5]))  # <- 你预估的零点
    plt.plot(x_, func1(x_))
    plt.hlines(0, -10, 10)
    plt.scatter(min_global[0], func1(min_global[0]), label='global min')
    plt.scatter(min_local, func1(np.float(min_local.item())), label='local min')
    plt.scatter(root.x, func1(root.x), label='roots')
    plt.legend(loc='best')
    plt.show()


plot1()
