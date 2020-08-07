import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = np.genfromtxt('wine_data.csv', delimiter=',')

xData = data[:, 1:]
yData = data[:, 0]
# print(xData, '\n\n', yData)
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData)
scalar = StandardScaler()
xTrain = scalar.fit_transform(xTrain)
xTest = scalar.fit_transform(xTest)

mlp = MLPClassifier(max_iter=500)
mlp.fit(xTrain, yTrain)

yPredict = mlp.predict(xTest)

print(classification_report(yTest, yPredict))
print('\n------------------------------------------------------------------------------------------------------\n')
print(confusion_matrix(yTest, yPredict))
