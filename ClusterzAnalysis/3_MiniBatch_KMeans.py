"""

MiniBatch
    采用小批量数据子集减少计算时间
    虽略差于标准算法，但大大缩短时间，适合大量数据
    随机抽取数据形成小批量，给他们分配最近的质心，再更新质心

"""
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

dataset = np.genfromtxt('kMeans.txt', delimiter=' ')

K = 4

mbModel = cluster.MiniBatchKMeans(n_clusters=4)
mbModel.fit(dataset)
center = mbModel.cluster_centers_
print('Centers:\n', center)
result = mbModel.labels_
marker = ['or', 'ob', 'og', 'oy']
center_marker = ['*r', '*b', '*g', '*y']
for index, data in enumerate(dataset):
    plt.plot(data[0], data[1], marker[result[index]])
for index, data in enumerate(center):
    plt.plot(data[0], data[1], center_marker[index], markersize=20)

# plt.show()
xMin, xMax = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
yMin, yMax = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
zPredict = mbModel.predict(np.c_[xx.flatten(), yy.flatten()])
zPredict = zPredict.reshape(xx.shape)
plt.contourf(xx, yy, zPredict, cmap='Pastel1')
plt.show()
