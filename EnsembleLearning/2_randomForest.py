"""

随机森林
    决策树+Bagging+随机属性选择

    bagging抽取样本随机
    从所有属性中选择建立决策树的属性随机
    重复上述过程，最后投票

"""
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('LR-testSet2.csv', delimiter=',')

xData = data[:, :-1]
yTarget = data[:, -1]

plt.scatter(xData[:, 0], xData[:, 1], c=yTarget)
plt.show()

xTrain, xTest, yTrain, yTest = train_test_split(xData, yTarget, test_size=0.5)


def plot(model, x_data, y_data):
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    z_predict = model.predict(np.c_[xx.flatten(), yy.flatten()])
    z_predict = z_predict.reshape(xx.shape)
    plt.contourf(xx, yy, z_predict, cmap='Pastel2')
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
    plt.show()


decisionTreeModel = tree.DecisionTreeClassifier()
decisionTreeModel.fit(xTrain, yTrain)
plot(decisionTreeModel, xData, yTarget)
print('Decision Tree Score:', decisionTreeModel.score(xTest, yTest))

randomForestModel = RandomForestClassifier(n_estimators=50)  # 随机样本50个
randomForestModel.fit(xTrain, yTrain)
plot(randomForestModel, xData, yTarget)
print('Random Forest Score:', randomForestModel.score(xTest, yTest))
