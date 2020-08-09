"""

AdaBoosting
    Adaptive Boosting 自适应增强
    将学习器的重点放在“容易”出错的样本上，可以提升学习器的性能
    上一次被错误分类的样本，下次的权值会增大
    而且，效率好的弱分类器在最终分类器中所占的权重越大

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

x1, y1 = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
# set1 = set(y1)
# print(x1[0], set1)
x2, y2 = make_gaussian_quantiles(n_samples=400, mean=(3, 3), n_features=2, n_classes=2)
# print(x2[0])

xData = np.concatenate((x1, x2))
yData = np.concatenate((y1, -y2 + 1))
# plt.scatter(xData[:, 0], xData[:, 1], c=yData)


decisionTreeModel = DecisionTreeClassifier(max_depth=3)
decisionTreeModel.fit(xData, yData)
x_min, x_max = xData[:, 0].min() - 1, xData[:, 0].max() + 1
y_min, y_max = xData[:, 1].min() - 1, xData[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
z_predict = decisionTreeModel.predict(np.c_[xx.flatten(), yy.flatten()])
z_predict = z_predict.reshape(xx.shape)
plt.contourf(xx, yy, z_predict, cmap='Pastel2')
plt.scatter(xData[:, 0], xData[:, 1], c=yData)
plt.show()
print('Decision Tree Score', decisionTreeModel.score(xData, yData))

adaBoostModel = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=10)
adaBoostModel.fit(xData, yData)
x_min, x_max = xData[:, 0].min() - 1, xData[:, 0].max() + 1
y_min, y_max = xData[:, 1].min() - 1, xData[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
z_predict = adaBoostModel.predict(np.c_[xx.flatten(), yy.flatten()])
z_predict = z_predict.reshape(xx.shape)
plt.contourf(xx, yy, z_predict, cmap='Pastel2')
plt.scatter(xData[:, 0], xData[:, 1], c=yData)
plt.show()
print('AdaBoost Score:', adaBoostModel.score(xData, yData))
