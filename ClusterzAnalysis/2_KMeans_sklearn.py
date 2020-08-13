from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

dataset = np.genfromtxt('kMeans.txt', delimiter=' ')

K = 4

k_means_model = cluster.KMeans(n_clusters=K)
k_means_model.fit(dataset)
center = k_means_model.cluster_centers_
print('Centers:\n', center)
results = k_means_model.labels_

marker = ['or', 'ob', 'og', 'oy']
center_marker = ['*r', '*b', '*g', '*y']
for index, data in enumerate(dataset):
    plt.plot(data[0], data[1], marker[results[index]])
for index, data in enumerate(center):
    plt.plot(data[0], data[1], center_marker[index], markersize=20)
# plt.show()

xMin, xMax = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
yMin, yMax = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
zPredict = k_means_model.predict(np.c_[xx.flatten(), yy.flatten()])
zPredict = zPredict.reshape(xx.shape)
plt.contourf(xx, yy, zPredict, cmap='Pastel1')
plt.show()
