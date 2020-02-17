"""
岭回归
可以抵消多重共线性
抗“病态矩阵”
限制系数平方和
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

data = np.genfromtxt("longley.csv", delimiter=",")  # <- 只能读数字, 字符是 nan

# cut
xData = data[1:, 2:]
yData = data[1:, 1]
print(xData)
print(yData)

alphasTest = np.linspace(0.001, 1, 50)
model = lm.RidgeCV(alphas=alphasTest, store_cv_values=True)  # CV -> Cross Verify
model.fit(xData, yData)

print(model.alpha_)  # 岭系数
print(model.cv_values_.shape)

plt.plot(alphasTest, model.cv_values_.mean(axis=0))
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)), "ro")
plt.show()

print(model.predict(xData))
