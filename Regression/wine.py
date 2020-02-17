import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = np.genfromtxt('./linear.csv', delimiter=',')

xData = data[1:, 0]
yData = data[1:, 1]
plt.scatter(xData, yData)
plt.title("Time and Quality Data")
plt.xlabel("Time")
plt.ylabel('Quality')
plt.show()

xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size=0.3)

xTrain = xTrain[:, np.newaxis]
xTest = xTest[:, np.newaxis]

model = LinearRegression()
model.fit(xTrain, yTrain)
plt.scatter(xTrain, yTrain, c='b')
plt.plot(xTrain, model.predict(xTrain), c='r')
plt.title('Time and Quality Train Predict')
plt.show()

plt.scatter(xTest, yTest, c='b')
plt.plot(xTest, model.predict(xTest), c='r')
plt.title('Time and Quality Test Predict')
plt.show()
print(model.score(xTest, yTest))
