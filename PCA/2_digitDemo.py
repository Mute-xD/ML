from sklearn import datasets
from sklearn import neural_network
from sklearn import model_selection
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from DimReduce import DimReduce


digits = datasets.load_digits()
xData = digits.data
yTarget = digits.target
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(xData, yTarget)

mlp = neural_network.MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
mlp.fit(xTrain, yTrain)
print(mlp.score(xTest, yTest))

dimReduce = DimReduce(xData, 2, debugMode=False)
plt.scatter(dimReduce.targetData[:, 0], dimReduce.targetData[:, 1], c=yTarget, cmap='Set3')
plt.show()

dimReduce = DimReduce(xData, 3, debugMode=False)
axis = plt.figure().add_subplot(projection='3d')
axis.scatter(dimReduce.targetData[:, 0], dimReduce.targetData[:, 1], dimReduce.targetData[:, 2], c=yTarget, cmap='Set3')
plt.show()
