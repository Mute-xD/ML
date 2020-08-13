"""

肘部法则
    随着K值的增加，costFunc会逐渐降低，寻找costFunc的突变点，我们称之为肘部，此处K表现较好
    但不一定找得到，找到了也不一定是.....

"""
import costFunc
import numpy as np
import matplotlib.pyplot as plt


class ElbowKMeans(costFunc.OptiKMeans):
    def functionSequence(self):
        distance_list = []
        for k in range(2, 10):
            self.K = k
            self.initCenter()
            self.isChanged = True
            self.runner()
            distance_list.append(sum(self.sampleAttr[:, 1]))
        print(distance_list)
        plt.plot(list(range(2, 10)), distance_list)
        plt.show()


datafile = np.genfromtxt('kMeans.txt', delimiter=' ')
opti = ElbowKMeans(datafile)
