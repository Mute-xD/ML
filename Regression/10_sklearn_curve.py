"""
sklearn实现
多项式回归
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = np.genfromtxt("job.csv", delimiter=",")
xData = data[1:, 1]
yData = data[1:, 2]
plt.scatter(xData, yData)
plt.show()
# 线性拟合
xData = np.atleast_2d(xData).T
yData = np.atleast_2d(yData).T
model = LinearRegression()
model.fit(xData, yData)
plt.plot(xData, yData, "b.")
plt.plot(xData, model.predict(xData), "r")
plt.show()

model = PolynomialFeatures(degree=10)  # 定义特征维度
xPoly = model.fit_transform(xData)  # 特征处理(升维)
lin_reg = LinearRegression()  # 对象声明
lin_reg.fit(xPoly, yData)

plt.plot(xData, yData, "b.")
xTest = np.atleast_2d(np.linspace(1, 10, 100)).T  # 高密度取样（平滑）
plt.plot(xTest, lin_reg.predict(model.fit_transform(xTest)), c="r")
plt.show()
print(lin_reg.coef_)
print(lin_reg.intercept_)
