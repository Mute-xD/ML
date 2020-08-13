"""

DBSCAN
    基于密度的聚类算法
    将具有足够高密度的区域划分为簇
    可发现任何形状的聚类

    定义：
        ε邻域：给定对象半径ε内的区域
        核心对象：如果给定ε邻域内的样本点数大于等于MinPoints，则该对象为核心对象
        直接密度可达：给定一个对象集合D，如果p在q的𝜀邻域内，且q是一个核心对象，则对象p从q出发是直接密度可达的
        密度可达：集合D，存在一个对象链p1,p2…pn,p1=q,pn=p,pi+1是从pi关于𝜀和MinPoints直接密度可达，则称点p是从q关于𝜀和MinPoints密度可达
        密度相连：集合D存在点o，使得点p、q是从o关于𝜀和MinPoints密度可达的，那么点p、q是关于𝜀和MinPoints密度相连的

    实现：
        1. 指定ε和MinPoints
        2. 计算所有样本点，若点P的ε邻域中存在超过MinPoints个点，则创建一个以P为核心点的新簇
        3. 反复寻找这些核心点直接密度可达（之后可能是密度可达）的点，将其加入到相应的簇，对于核心点发生“密度相连”的簇，给予合并
        4. 当没有新的点可以被添加到任何簇时，算法结束

    缺点：
        当数据量增大时，要求较大的内存支持I/O消耗也很大
        当空间聚类的密度不均匀、聚类间距差相差很大时，聚类质量较差

    与K-Means相比：
        DBSCAN不需要输入聚类个数
        聚类簇的形状没有要求
        可以在需要时输入过滤噪声的参数（MinPoints，ε）

"""
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

dataset = np.genfromtxt('kMeans.txt', delimiter=' ')

dbModel = cluster.DBSCAN(eps=1.5, min_samples=4)
dbModel.fit(dataset)
result = dbModel.labels_  # label=-1 是噪声

marker = ['or', 'ob', 'og', 'oy', 'ok', 'om']
for index, data in enumerate(dataset):
    plt.plot(data[0], data[1], marker[result[index]])
    # plt.scatter(data[0], data[1], c=result[index])
plt.show()
