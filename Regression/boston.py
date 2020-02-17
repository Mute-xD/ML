from sklearn.datasets import load_boston
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

house = load_boston()
# print(house.DESCR)
x = house.data
y = house.target
df = pd.DataFrame(x, columns=house.feature_names)
df["Target"] = pd.DataFrame(y, columns=["Target"])
print(df.head())

plt.figure(figsize=(15, 15))
p = sns.heatmap(df.corr(), annot=True, square=True)
# plt.show()

# 数据标准化 （化为0附近的值）
std = StandardScaler()
x = std.fit_transform(x)
print(x[:5])

# 切分数据 训练集和测试集
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)

model = LassoCV(cv=3)
model.fit(xTrain, yTrain)

print(model.alpha_)  # lasso 系数
print(model.coef_)  # 相关系数
print(model.score(xTest, yTest))  # 回归系数
