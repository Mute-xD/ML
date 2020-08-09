"""
LASSO
可以压缩一些系数，强制系数绝对值之和小于某定值，同时设定一些回归系数为零
当 λ 充分大时，可以把某些待估系数精确地收缩到零
"""
import numpy as np
import sklearn.linear_model as lm

data = np.genfromtxt("longley.csv", delimiter=",")
xData = data[1:, 2:]
yData = data[1:, 1]
print(xData.shape, yData.shape)

model = lm.LassoCV(cv=5)  # 要求预设参数cv = 3~5
model.fit(xData, yData)
print(model.alpha_)
print(model.coef_)
print(model.predict(np.atleast_2d(xData)))
