# import random
import operator
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def knn(x_test, x_data, y_data, k):
    x_data_size = x_data.shape[0]
    distance = ((((np.tile(x_test, (x_data_size, 1))) - x_data) ** 2).sum(axis=1)) ** 0.5  # 欧几里得距离实现
    sorted_distance = distance.argsort()
    class_count = {}
    class_count = class_count.fromkeys(y_data, 0)
    for j in range(k):
        voted_label = y_data[sorted_distance[j]]
        class_count[voted_label] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


iris = datasets.load_iris()
xTrain, xTest, yTrain, yTest = train_test_split(iris.data, iris.target, train_size=0.2)  # 数据切分，分割比0.2（test 0.2）
# # 数据切分实现 Start
# dataSize = iris.data.shape[0]
# index = [i for i in range(dataSize)]
# random.shuffle(index)  # 打乱
# iris.data = iris.data[index]  # 根据乱序索引重定向
# iris.target = iris.target[index]
#
# testSize = 40
# xTrain = iris.data[testSize:]  # 40 - END
# xTest = iris.data[:testSize]  # 0 - 40
# yTrain = iris.target[testSize:]
# yTest = iris.target[:testSize]
# # 数据切分实现 End

prediction = []
for i in range(xTest.shape[0]):
    prediction.append(knn(xTest[i], xTrain, yTrain, 3))
print(classification_report(yTest, prediction))

print(confusion_matrix(yTest, prediction))  # 混淆矩阵
