from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("data.csv", delimiter=",")
xData = data[:, 0]
yData = data[:, 1]
plt.plot(xData, yData, "b.")
plt.show()
print(xData.shape)  # (100,)

xData = np.atleast_2d(xData).T
print(xData.shape)  # (100,1)
yData = np.atleast_2d(yData).T
model = LinearRegression()  # 创建对象
model.fit(xData, yData)  # 接口类型:(N, 1)

plt.plot(xData, yData, "b.")
plt.plot(xData, model.predict(xData), "r")
plt.show()
