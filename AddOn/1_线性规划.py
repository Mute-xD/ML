"""

Liner Program
    线性规划
    用于在给定约束条件下最大限度地改善指定的指标
    所有的图算法都可使用线性规划来实现
    Simplex
        单纯形算法
        可行域大多是凸多胞体，在这样的可行域上求解问题的最优解，都会出现在顶点的位置
        我们求解最优解的过程转化为在顶点中寻找最优解的过程

    scipy的简单模型，复杂模型需要使用pulp构建
    https://www.jianshu.com/p/9be417cbfebb

"""
import numpy as np
from scipy import optimize

"""
          min z = 2x1 + 3x2 + x3
          
x1 + 4x2 + 2x3 >= 8
     3x1 + 2x2 >= 6
x1, x2, x3 >= 0
"""
z = np.array([2, 3, 1])
a = np.array([[1, 4, 2], [3, 2, 0]])
b = np.array([8, 6])
x1_area = x2_area = x3_area = (0, None)
answer = optimize.linprog(z, A_ub=-a, b_ub=-b, bounds=(x1_area, x2_area, x3_area))
print(answer)
