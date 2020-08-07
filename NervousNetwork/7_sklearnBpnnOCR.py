from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


digit = load_digits()
scalar = StandardScaler()

xData = scalar.fit_transform(digit.data)
yLabel = digit.target
xTrain, xTest, yTrain, yTest = train_test_split(xData, yLabel)
# 调包一时爽，一直调包一直爽
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
mlp.fit(xTrain, yTrain)
prediction = mlp.predict(xTest)
print(classification_report(yTest, prediction))
print('-----------------------------------------------------------------------------------------------------')
print(confusion_matrix(yTest, prediction))
