"""
集成学习
装袋
    也叫 Bootstrap Aggregating
    在原始数据集选择S次后得到S个新数据集，是一种有放回抽样

"""
from sklearn import neighbors
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

irisDataset = datasets.load_iris()
xData = irisDataset.data[:, :2]  # 为了强行拉低准确率，这里就用了两个特征，而且特征多了画不了图（高维）
yTarget = irisDataset.target


def plot(model, x_data, y_data):
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),  # TODO: 把高维的mesh实现了，画图的时候再单独选两个XY
                         np.arange(y_min, y_max, 0.02))  # 别用x_ y_ ，画图前先用x0,x1,x2.x3表示
    z_predict = model.predict(np.c_[xx.flatten(), yy.flatten()])
    z_predict = z_predict.reshape(xx.shape)
    plt.contourf(xx, yy, z_predict, cmap='Pastel2')
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
    plt.show()


xTrain, xTest, yTrain, yTest = train_test_split(xData, yTarget)

knnModel = neighbors.KNeighborsClassifier()
knnModel.fit(xTrain, yTrain)

plot(knnModel, xData, yTarget)
print('KNN Score:', knnModel.score(xTest, yTest))

decisionTreeModel = tree.DecisionTreeClassifier()
decisionTreeModel.fit(xTrain, yTrain)

plot(decisionTreeModel, xData, yTarget)
print('DT  Score:', decisionTreeModel.score(xTest, yTest))

baggingKNN = BaggingClassifier(knnModel, n_estimators=100)  # 传一个模型  100次有放回抽样
baggingKNN.fit(xTrain, yTrain)
plot(baggingKNN, xData, yTarget)
print('baggingKNN Score', baggingKNN.score(xTrain, yTrain))
