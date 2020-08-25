"""

求函数的导数
    http://liao.cpython.org/scipy17/

"""

import numpy as np
import scipy.misc
import sympy


def func(x):
    return x ** 5


for i in range(1, 4):
    print(scipy.misc.derivative(func, i, dx=1e-6, n=2))  # 1, 2, 3, 4的二阶导数

##################################################################################

t = sympy.symbols('x')
for i in range(1, 4):
    print(sympy.diff(t ** 5, t, i))  # 对t求i阶导数
    print(sympy.diff(t ** 5, t, i).subs(t, i), i)  # 代入

###################################################################################

poly = np.poly1d([1, 0, 0, 0, 0, 0])
print(poly)
for i in range(1, 4):
    print(np.polyder(poly, i))
    print(np.polyder(poly, i)(1.0))
# or
for i in range(1, 4):
    print(poly.deriv(i))
    print(poly.deriv(i)(1.0))
