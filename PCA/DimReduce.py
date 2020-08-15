"""

PCA
    主成分分析
    降维，利于可视化
    寻找数据最重要的方向（方差最大方向）
    第一个主成分就是从数据差异性最大(方差最大)的方向提取出来的，第二个主成分则来自于数据差异性次大的方向，并且要与第一个主成分方向正交
    方法：
        1. 预处理 数据中心化 -> x - xBar
        2. 求样本协方差矩阵 1/m * x * x.T
        3. 对协方差矩阵进行特征分解
        4. 选出最大的K个特征值对应的K个特征向量
        5. 将原始数据投影到选取的特征向量上
        6. 输出结果

    这个轮子貌似还挺实用的

"""
import numpy as np
import matplotlib.pyplot as plt


class DimReduce:
    def __init__(self, data, targetDim, debugMode=False):
        self.originData = data
        self.targetDim = targetDim
        self.meanData = None
        self.covMat = None
        self.eigenVal = None
        self.eigenVector = None
        self.chosenEigVector = None
        self.meanValue = None
        self.debugMode = debugMode
        self.targetData = None
        self.functionSequence()

    def functionSequence(self):
        self.zeroMeans()
        self.setCovMat()
        self.setEigenMat()
        self.convert()

    def zeroMeans(self):
        self.meanValue = np.mean(self.originData, axis=0)
        self.meanData = self.originData - self.meanValue

    def setCovMat(self):  # 协方差矩阵
        self.covMat = np.cov(self.meanData, rowvar=False)  # 一行一样本
        if self.debugMode:
            print('Covariance:\n', self.covMat)
        return self.covMat

    def setEigenMat(self):
        self.eigenVal, self.eigenVector = np.linalg.eig(self.covMat)
        if self.debugMode:
            print('EigenVal ->', self.eigenVal)
            print('EigenMatrix:\n', self.eigenVector)

    def convert(self):
        eig_val_sort = np.argsort(self.eigenVal)
        chosen_eig_val = eig_val_sort[-1:-(self.targetDim + 1):-1]  # 反着取targetDim个eigVal
        chosen_eig_vector = np.mat(self.eigenVector[:, chosen_eig_val])
        self.chosenEigVector = chosen_eig_vector
        if self.debugMode:
            print('Chosen EigenMatrix:\n', chosen_eig_vector)
        self.targetData = np.array(self.meanData * chosen_eig_vector)
        return self.targetData

    def reform(self):
        return (self.targetData * self.chosenEigVector.T) + self.meanValue


if __name__ == '__main__':
    dataset = np.genfromtxt('data.csv', delimiter=',')
    xData = dataset[:, 0]
    yData = dataset[:, 1]
    plt.scatter(xData, yData)
    dimReduce = DimReduce(dataset, 1, debugMode=True)
    # plt.scatter(np.arange(dimReduce.targetData.size), dimReduce.targetData)  # 一维点
    reformed = np.array(dimReduce.reform())
    plt.scatter(reformed[:, 0], reformed[:, 1])
    plt.show()
