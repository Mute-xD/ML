"""

SVM解决非线性问题
    把低维的非线性问题，映射到高维成为线性问题

"""
import matplotlib.pyplot as plt
from sklearn import datasets
# from mpl_toolkits import mplot3d

xData, yData = datasets.make_circles(n_samples=500, factor=0.3, noise=0.1)
plt.scatter(xData[:, 0], xData[:, 1], c=yData, cmap='tab10')
plt.show()
zData = xData[:, 0] ** 2 + xData[:, 1] ** 2
axis = plt.figure().add_subplot(projection='3d')
axis.scatter(xData[:, 0], xData[:, 1], zData, c=yData, cmap='tab10')
plt.show()
