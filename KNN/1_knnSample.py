import numpy as np
import matplotlib.pyplot as plt
import operator

# k值定义
k = 3

x0 = np.array([3, 2, 1])
y0 = np.array([104, 100, 81])
x1 = np.array([101, 99, 98])
y1 = np.array([10, 5, 2])
scatter1 = plt.scatter(x0, y0, c='r')
scatter2 = plt.scatter(x1, y1, c='b')

xUnknown = np.array([[18, 90]])
scatter3 = plt.scatter(xUnknown[0, 0], xUnknown[0, 1], c='k')
plt.legend(handles=[scatter1, scatter2, scatter3], labels=['A', 'B', 'X'], loc='best')
# plt.show()

xData = np.r_[np.c_[x0, y0], np.c_[x1, y1]]
yData = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
xUnknown = np.tile(xUnknown, (xData.shape[0], 1))  # 贴瓷砖
distance = (((xUnknown - xData) ** 2).sum(axis=1)) ** 0.5  # 欧几里得距离实现
sortedDistance = distance.argsort()  # 对索引排序
# print(sortedDistance)

classCount = {}
classCount = classCount.fromkeys(yData, 0)  # 初始化
for i in range(k):
    votedLabel = yData[sortedDistance[i]]
    classCount[votedLabel] += 1
    # classCount[votedLabel] = classCount.get(votedLabel, 0) + 1

classCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 针对 Keys 排序
bestMatchClass = classCount[0][0]
print(bestMatchClass)
