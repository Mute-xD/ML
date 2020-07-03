import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import linear_model

scale = True  # 标准化开关

data = np.genfromtxt("./LR-testSet.csv", delimiter=",")
xData = data[:, :-1]
yData = data[:, -1]


def plot():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for i in range(len(xData)):
        if yData[i] == 0:
            x0.append(xData[i, 0])
            y0.append(xData[i, 1])
        else:
            x1.append(xData[i, 0])
            y1.append(xData[i, 1])

    scatter0 = plt.scatter(x0, y0, c="b", marker="o")
    scatter1 = plt.scatter(x1, y1, c="r", marker="x")

    plt.legend(handles=[scatter0, scatter1], labels=["label0", "label1"], loc="best")


plot()
plt.show()

if scale:
    xData = preprocessing.scale(xData)

lr = linear_model.LogisticRegression()
lr.fit(xData, yData)
print(lr.intercept_, "    ", lr.coef_[0][0], "    ", lr.coef_[0][1])

plot()
xExample = np.array([[-4], [3]])  # 预先指定回归线左右x坐标
yExample = (-lr.intercept_ - xExample * lr.coef_[0][0]) / lr.coef_[0][1]
plt.plot(xExample, yExample, "k")
plt.show()

prediction = lr.predict(xData)
print(classification_report(yData, prediction))  # predict 和 yData 切合度
