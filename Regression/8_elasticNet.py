"""
弹性网
正则项中参数次幂可变
可理解为 岭回归 LASSO 的组合
排除多重共线性
"""
import numpy as np
import sklearn.linear_model as lm

data = np.genfromtxt("longley.csv", delimiter=",")
xData = data[1:, 2:]
yData = data[1:, 1]

model = lm.ElasticNetCV(cv=3)
model.fit(xData, yData)
print(model.alpha_)
print(model.coef_)
print(model.predict(xData))
