import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import spatial
from sko.DE import DE
from sko.GA import GA
from sko.GA import GA_TSP
from sko.PSO import PSO
from sko.SA import SA
from sko.SA import SA_TSP
from sko.ACA import ACA_TSP
from sko.IA import IA_TSP
from sko.AFSA import AFSA

"""
scikit-opt
    https://scikit-opt.github.io/scikit-opt/#/zh/README
    一个封装了7种启发式算法的 Python 代码库
    差分进化算法、遗传算法、粒子群算法、模拟退火算法、蚁群算法、鱼群算法、免疫优化算法
    参数表：
        https://scikit-opt.github.io/scikit-opt/#/zh/args

"""

'''
# 差分进化算法
例子是规划
min f(x1, x2, x3) = x1^2 + x2^2 + x3^2
s.t.
    x1*x2 >= 1
    x1*x2 <= 5
    x2 + x3 = 1
    0 <= x1, x2, x3 <= 5
'''


def func1(p):
    x1, x2, x3 = p
    return x1 ** 2 + x2 ** 2 + x3 ** 2


constraintEqual = [lambda x: 1 - x[1] - x[2]]  # x2 + x3 = 1
constraintUnequal = [lambda x: 1 - x[0] * x[1],  # x1*x2 >= 1
                     lambda x: x[0] * x[1] - 5]  # x1*x2 <= 5
de = DE(func=func1, n_dim=3, size_pop=50, max_iter=800, lb=[0, 0, 0], ub=[5, 5, 5],
        constraint_eq=constraintEqual, constraint_ueq=constraintUnequal)
best_x, best_y = de.run()
print('差分进化\n')
print('best_x:', best_x, '\n', 'best_y:', best_y)

'''
遗传算法
    这里的func有一堆局部最小值，全局最小值在（0， 0），是0
'''


def func2(p):  # 涟漪状
    x1, x2 = p
    x = np.square(x1) + np.square(x2)
    return 0.5 + (np.sin(x) - 0.5) / np.square(1 + 0.001 * x)


ga = GA(func=func2, n_dim=2, size_pop=50, max_iter=200, lb=[-1, -1], ub=[1, 1], precision=1e-7)
best_x, best_y = ga.run()
print('\n遗传\n')
print('best_x:', best_x, '\nbest_y:', best_y)

# Y_history = pd.DataFrame(ga.all_history_Y)
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
# Y_history.min(axis=1).cummin().plot(kind='line')
# plt.show()

'''
遗传算法用于旅行商问题
'''
NUM_POINT = 50
pointsCoordinate = np.random.rand(NUM_POINT, 2)
distanceMatrix = spatial.distance.cdist(pointsCoordinate, pointsCoordinate, metric='euclidean')


def getTotalDistance(routine):
    num_points, = routine.shape
    return sum([distanceMatrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


ga_tsp = GA_TSP(func=getTotalDistance, n_dim=NUM_POINT, size_pop=50, max_iter=500, prob_mut=1)
best_points, best_distance = ga_tsp.run()
print('\n遗传旅行商\n')
print(best_points, '\nbest Distance:', best_distance)

# fig, ax = plt.subplots(1, 2)
# best_points_ = np.concatenate([best_points, [best_points[0]]])
# best_points_coordinate = pointsCoordinate[best_points_, :]
# ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
# ax[1].plot(ga_tsp.generation_best_Y)
# plt.show()
'''
粒子群算法
'''


def func3(x):
    x1, x2, x3 = x
    return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2


pso = PSO(func=func3, dim=3, pop=40, max_iter=150, lb=[0, -1, 0.5], ub=[1, 1, 1], w=0.8, c1=0.5, c2=0.5)
pso.run()
print('\n带约束的粒子群算法\n')
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
# plt.plot(pso.gbest_y_hist)
# plt.show()
pso = PSO(func=func3, dim=3)
pso.run()
print('\n不带约束的粒子群算法\n')
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
'''
模拟退火算法
'''
sa = SA(func=func3, x0=[1, 1, 1], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
best_x, best_y = sa.run()
print('\n模拟退火算法用于多元函数优化\n')
print('best_x:', best_x, 'best_y', best_y)
# plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
# plt.show()
sa_tsp = SA_TSP(func=getTotalDistance, x0=range(NUM_POINT), T_max=100, T_min=1, L=10 * NUM_POINT)
best_points, best_distance = sa_tsp.run()
print('\n模拟退火算法解决TSP问题（旅行商问题）\n')
print(best_points, '\nBest Distance:', best_distance)

# from matplotlib.ticker import FormatStrFormatter
# fig, ax = plt.subplots(1, 2)
# best_points_ = np.concatenate([best_points, [best_points[0]]])
# best_points_coordinate = pointsCoordinate[best_points_, :]
# ax[0].plot(sa_tsp.best_y_history)
# ax[0].set_xlabel("Iteration")
# ax[0].set_ylabel("Distance")
# ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
#            marker='o', markerfacecolor='b', color='r', linestyle='-')
# ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax[1].set_xlabel("Longitude")
# ax[1].set_ylabel("Latitude")
# plt.show()
'''
蚁群算法(好慢)
'''
aca = ACA_TSP(func=getTotalDistance, n_dim=NUM_POINT, size_pop=50, max_iter=200, distance_matrix=distanceMatrix)
best_points, best_distance = aca.run()
print('\n蚁群算法\n')
print('best_points:', best_points, '\nbest_distance', best_distance)
'''
免疫优化算法(也好慢)
'''
ia_tsp = IA_TSP(func=getTotalDistance, n_dim=NUM_POINT, size_pop=500, max_iter=800, prob_mut=0.2, T=0.7, alpha=0.95)
best_points, best_distance = ia_tsp.run()
print('\n免疫优化算法\n')
print('best routine:', best_points, '\nbest_distance:', best_distance)
'''
人工鱼群算法
'''


def func4(x):
    x1, x2 = x
    return 1 / x1 ** 2 + x1 ** 2 + 1 / x2 ** 2 + x2 ** 2


afsa = AFSA(func4, n_dim=2, size_pop=50, max_iter=300, max_try_num=100, step=0.5, visual=0.3, q=0.98, delta=0.5)
best_x, best_y = afsa.run()
print('\n人工鱼群算法\n')
print('best_x:', best_x, '\nbest_y:', best_y)
