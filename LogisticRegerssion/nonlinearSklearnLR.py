import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import PolynomialFeatures

xData, yData = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)
# samples 样本数 features 特征数 classes 分类数
plt.scatter(xData[:, 0], xData[:, 1], c=yData)
# plt.show()

# linear start
logistic = linear_model.LogisticRegression()
logistic.fit(xData, yData)

xMin, xMax = xData[:, 0].min() - 1, xData[:, 0].max() + 1
yMin, yMax = xData[:, 1].min() - 1, xData[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))

z = logistic.predict(np.c_[xx.ravel(),  # 扁平化 predict
                           yy.ravel()])
z = z.reshape(xx.shape)
cs = plt.contourf(xx, yy, z)
plt.scatter(xData[:, 0], xData[:, 1], c=yData)
plt.show()
print("score(linear):    ", logistic.score(xData, yData))
print("线性系数：\n", logistic.coef_)
# linear end


# nonlinear start
polyRegression = PolynomialFeatures(degree=2)  # 阶数，多试几个
xPoly = polyRegression.fit_transform(xData)
logistic = linear_model.LogisticRegression()
logistic.fit(xPoly, yData)
xMin, xMax = xData[:, 0].min() - 1, xData[:, 0].max() + 1
yMin, yMax = xData[:, 1].min() - 1, xData[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),  # 取样密度
                     np.arange(yMin, yMax, 0.02))

z = logistic.predict(polyRegression.fit_transform(np.c_[xx.ravel(),
                                                        yy.ravel()]))
z = z.reshape(xx.shape)
cs1 = plt.contourf(xx, yy, z)
plt.scatter(xData[:, 0], xData[:, 1], c=yData)
plt.show()
print("score(nonlinear):    ", logistic.score(xPoly, yData))
print("多项式系数：\n", logistic.coef_)
print("多项式阶数：    ", polyRegression.degree)
# nonlinear end
