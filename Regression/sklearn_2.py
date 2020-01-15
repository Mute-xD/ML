import numpy as np
from sklearn import linear_model

data = np.genfromtxt("Delivery.csv", delimiter=",")
print(data)
xData = data[:, 0:2]
yData = data[:, 2]
print(xData)
print(yData)

model = linear_model.LinearRegression()
model.fit(xData, yData)

print("coefficients", model.coef_)  # 系数
print("intercept", model.intercept_)  # 截距
xTest = [[102, 4]]
print("predict", model.predict(xTest))
