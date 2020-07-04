import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

scale = False
data = np.genfromtxt("LR-testSet2.csv", delimiter=",")
xData = data[:, 0:2]
yData = data[:, 2]

if scale:
    xData = preprocessing.scale(xData)


def plot():
    x0, x1, y0, y1 = [], [], [], []
    for i in range(len(xData)):
        if not yData[i]:
            x0.append(xData[i, 0])
            y0.append(xData[i, 1])
        if yData[i]:
            x1.append(xData[i, 0])
            y1.append(xData[i, 1])
    scatter0 = plt.scatter(x0, y0, c="b", marker="o")
    scatter1 = plt.scatter(x1, y1, c="r", marker="x")
    plt.legend(handles=[scatter0, scatter1], labels=["label0", "label1"], loc="best")


plot()
# plt.show()

polyRegression = PolynomialFeatures(degree=3)
xPoly = polyRegression.fit_transform(xData)  # xPoly-> (偏置, x, y, x^2, x*y, y^2)degree=2生成二阶非线性项


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def cost(x_mat, y_mat, ws_):
    left = np.multiply(y_mat, np.log(sigmoid(x_mat * ws_)))
    right = np.multiply(1 - y_mat, np.log(1 - sigmoid(x_mat * ws_)))
    return np.sum(left + right) / -len(x_mat)


def gradAscent(x_arr, y_arr):
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr)
    lr = 0.03
    epochs = 50000
    cost_list = []
    m, n = np.shape(x_mat)
    ws_ = np.mat(np.ones((n, 1)))

    for i in range(epochs + 1):
        h = sigmoid(x_mat * ws_)
        ws_grid = x_mat.T * (h - y_mat.T) / m
        ws_ = ws_ - lr * ws_grid

        if i % 50 == 0:
            cost_list.append(cost(x_mat, y_mat, ws_))

    return ws_, cost_list


ws, cost_line = gradAscent(xPoly, yData)
print(ws)  # result

# 确定图形边缘（max+1, min-1）
xMin, xMax = xData[:, 0].min() - 1, xData[:, 0].max() + 1
yMin, yMax = xData[:, 1].min() - 1, xData[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
z = sigmoid(polyRegression.fit_transform(np.c_[xx.ravel(), yy.ravel()]).dot(np.array(ws)))
for j in range(len(z)):
    if z[j] > 0.5:
        z[j] = 1
    else:
        z[j] = 0
z = z.reshape(xx.shape)
cs = plt.contourf(xx, yy, z)  # 等高线图
plot()
# plt.show()


def predict(x_data, ws_):
    x_mat = np.mat(x_data)
    ws_ = np.mat(ws_)
    return [1 if x > 0.5
            else 0
            for x in sigmoid(x_mat * ws_)]


prediction = predict(xPoly, ws)
print(metrics.classification_report(yData, prediction))
