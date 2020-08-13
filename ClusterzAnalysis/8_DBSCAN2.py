import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cluster

x1, y1 = datasets.make_circles(n_samples=2000, factor=0.5, noise=0.05)
x2, y2 = datasets.make_blobs(n_samples=1000, centers=[[1.2, 1.2]], cluster_std=[[0.1]])

xData = np.concatenate((x1, x2))
# plt.scatter(xData[:, 0], xData[:, 1])
# plt.show()

kMeansModel = cluster.KMeans(n_clusters=3)
kMeansModel.fit(xData)
kmPredict = kMeansModel.labels_
plt.scatter(xData[:, 0], xData[:, 1], c=kmPredict)
plt.show()

dbModel = cluster.DBSCAN(eps=0.2, min_samples=50)
dbModel.fit(xData)
dbPredict = dbModel.labels_
plt.scatter(xData[:, 0], xData[:, 1], c=dbPredict)
plt.show()
