"""

聚类算法
    无监督学习
    K-Means
        K: 划分为K个聚类
        以空间中k个点为中心进行聚类，对最靠近他们的对象归类。通过迭代的方法，逐次更新各聚类中心的值，直至得到最好的聚类结果

    注意：造的这个轮子好像有点问题，不一定每次运行都能得到K个簇，建议多跑几次
"""
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, data, K=4):
        self.data = data
        self.K = K
        self.sampleCount = self.data.shape[0]
        self.dim = self.data.shape[1]
        self.sampleAttr = np.zeros((self.sampleCount, 2))  # 第一列：属于哪个聚类，第二列：与所属聚类的误差
        self.center = None
        self.isChanged = True
        self.functionSequence()

    def functionSequence(self):
        self.initCenter()
        self.runner()
        self.checker(autoRetry=True)
        self.showResult(isShow=False)
        self.plotClusterArea()

    @staticmethod
    def getDistance(a, b):
        return np.sqrt(sum((a - b) ** 2))  # a - b 平方开根号

    def initCenter(self):  # 随机选定重心
        self.center = np.zeros((self.K, self.dim))
        for i in range(self.K):
            chosen_index = np.random.randint(low=0, high=self.sampleCount)
            self.center[i, :] = self.data[chosen_index, :]

    def runner(self):
        while self.isChanged:
            self.isChanged = False
            for i in range(self.sampleCount):  # 样本循环
                min_distance = float('inf')
                best_fixed_cluster = 0
                for j in range(self.K):  # 每一个簇循环
                    distance = self.getDistance(self.center[j, :], self.data[i, :])
                    if distance < min_distance:
                        min_distance = distance
                        self.sampleAttr[i, 1] = distance
                        best_fixed_cluster = j
                if self.sampleAttr[i, 0] != best_fixed_cluster:  # 若样本的簇划分变化
                    self.isChanged = True
                    self.sampleAttr[i, 0] = best_fixed_cluster
            for j in range(self.K):
                cluster_index = np.nonzero(self.sampleAttr[:, 0] == j)  # 找到符合条件判定的索引
                points_in_cluster = self.data[cluster_index]  # 簇内的点
                self.center[j, :] = np.mean(points_in_cluster, axis=0)  # 重计算重心

    def showResult(self, data=None, isShow=True):  # data 若大于二维需要预处理
        if data is None:
            data = self.data
        dim = data.shape[1]
        if dim != 2:
            raise ValueError('dim of data can ONLY be 2')
        sample_color = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr']
        mark_color = ['*r', '*b', '*g', '*k', '^b', '+b', 'sb', 'db']
        for i in range(self.sampleCount):
            mark_index = int(self.sampleAttr[i, 0])
            plt.plot(data[i, 0], data[i, 1], sample_color[mark_index])
        for i in range(self.K):
            plt.plot(self.center[i, 0], self.center[i, 1], mark_color[i], markersize=20)
        if isShow:
            plt.show()

    def checker(self, autoRetry=False):
        if np.isnan(self.center).any():
            print('nan Found!\nRetry Needed')
            print(self.center)
            print('-------------------------------------------------------------')
            if autoRetry:
                self.isChanged = True
                self.functionSequence()
        else:
            print('-------------------------------------------------------------\n'
                  'Center:\n', self.center)
            print('Done!')

    def predict(self, xTest):
        x_test = np.atleast_2d(xTest)
        best_match_list = []
        for test in x_test:
            best_match = (np.argmin(((np.tile(test, (self.K, 1)) - self.center) ** 2).sum(axis=1)))
            best_match_list.append(best_match)
        return np.array(best_match_list)

    def plotClusterArea(self, data=None):
        if data is None:
            data = self.data
        dim = data.shape[1]
        if dim != 2:
            raise ValueError('dim of data can ONLY be 2')
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        z_predict = self.predict(np.c_[xx.flatten(), yy.flatten()])
        z_predict = z_predict.reshape(xx.shape)

        plt.contourf(xx, yy, z_predict, cmap='Pastel1')
        # plt.scatter(self.data[:, 0], self.data[:, 1])
        plt.show()


if __name__ == '__main__':
    _data = np.genfromtxt('kMeans.txt', delimiter=' ')
    k_means = KMeans(_data)
