"""

    若不加限制，线性决策树会出现过拟合
    DecisionTreeClassifier的参数：
        max_depth 最大深度，自己调
        min_sample_split 内部节点再划分所需的最大样本数
        criterion gini 基尼 entropy 信息熵

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics
import graphviz

data = np.genfromtxt('LR-testSet2.csv', delimiter=',')
xData = data[:, :-1]
yTarget = data[:, -1]
plt.scatter(xData[:, 0], xData[:, 1], c=yTarget, cmap='Set3')
plt.show()

xTrain, xTest, yTrain, yTest = model_selection.train_test_split(xData, yTarget)

# model = tree.DecisionTreeClassifier(criterion='gini')  # 出现了过拟合
model = tree.DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=4)
model.fit(xTrain, yTrain)

dotData = tree.export_graphviz(model, feature_names=['X', 'Y'], class_names=['A', 'B'],
                               filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dotData)
graph.render(format='svg')

yPredict = model.predict(xTest)
print(metrics.classification_report(yTest, yPredict))

xMax, xMin = xData[:, 0].max() + 1, xData[:, 0].min() - 1
yMax, yMin = xData[:, 1].max() + 1, xData[:, 1].min() - 1

xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
zPredicted = model.predict(np.c_[xx.flatten(), yy.flatten()])
zPredicted = zPredicted.reshape(xx.shape)

contourGraph = plt.contourf(xx, yy, zPredicted, cmap='Set3')
plt.scatter(xData[:, 0], xData[:, 1], c=yTarget, cmap='RdBu')
plt.show()
