"""

多次初始化，选取代价函数最小的质心作为聚类初始化结果
    代价函数直接用了getDistance，实际上应该去掉开方的
    还行把，貌似增大了出现聚类少于预期的情况的出现几率...
    面向对象真香

"""
import KMeans
import numpy as np


class OptiKMeans(KMeans.KMeans):
    def initCenter(self):
        best_center = None
        best_distance = float('inf')
        for _ in range(500):  # <- 随机生成质心的次数
            current_center = np.zeros((self.K, self.dim))
            distance = 0
            for k in range(self.K):
                chosen_index = np.random.randint(0, self.sampleCount)
                current_center[k, :] = self.data[chosen_index, :]
                distance += self.getDistance(self.data, current_center[k]).sum()
            distance /= self.K
            if distance < best_distance:
                best_distance = distance
                best_center = current_center
                # print('best ->', best_distance)
                # print('best center\n', best_center)
        print('Center Init:\n', best_center)
        self.center = best_center


if __name__ == '__main__':
    datafile = np.genfromtxt('kMeans.txt', delimiter=' ')
    opti = OptiKMeans(datafile)
