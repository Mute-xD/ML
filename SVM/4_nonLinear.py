import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm

dataset = np.genfromtxt('LR-testSet2.csv', delimiter=',')
xData = dataset[:, :-1]
yData = dataset[:, -1]


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


# plot()
# plt.show()
svmModel = svm.SVC(kernel='rbf')  # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' 核函数
svmModel.fit(xData, yData)
print(svmModel.score(xData, yData))

xMin, xMax = xData[:, 0].min() - 1, xData[:, 0].max() + 1
yMin, yMax = xData[:, 1].min() - 1, xData[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
zPredict = svmModel.predict(np.c_[xx.flatten(), yy.flatten()]).reshape(xx.shape)
plt.contourf(xx, yy, zPredict, cmap='tab10')
plot()
plt.show()
