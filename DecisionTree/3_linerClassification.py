import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz

data = np.genfromtxt('LR-testSet.csv', delimiter=',')
xData = data[:, :-1]
yTarget = data[:, -1]
print('xData:\n', xData, '\nyTarget:\n', yTarget)

plt.scatter(xData[:, 0], xData[:, 1], c=yTarget, cmap='Set3')
plt.show()

model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(xData, yTarget)

dotData = tree.export_graphviz(model, feature_names=['X', 'Y'], class_names=['A', 'B'],
                               filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dotData)
graph.render(format='svg')

# 开始画图
xMax, xMin = xData[:, 0].max() + 1, xData[:, 0].min() - 1
yMax, yMin = xData[:, 1].max() + 1, xData[:, 1].min() - 1

xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
zPredicted = model.predict(np.c_[xx.flatten(), yy.flatten()])
zPredicted = zPredicted.reshape(xx.shape)

contourGraph = plt.contourf(xx, yy, zPredicted, cmap='Set3')
plt.scatter(xData[:, 0], xData[:, 1], c=yTarget, cmap='RdBu')
plt.show()
