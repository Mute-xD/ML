from sklearn import tree
import numpy as np
import graphviz

data = np.genfromtxt('cart.csv', delimiter=',')  # 已经预先处理过，可以直接用numpy导入

xData = data[1:, 1:-1]
yTarget = data[1:, -1]
print('xData\n', xData, '\nyTarget\n', yTarget)

model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(xData, yTarget)

featureName = ['house_yes', 'house_no', 'single', 'married', 'divorced', 'income']
labelName = ['NO', 'YES']
dotData = tree.export_graphviz(model, feature_names=featureName, class_names=labelName,
                               filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dotData)
graph.render(format='svg')
