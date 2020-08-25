"""

熵值法确定信息权重
    https://www.jianshu.com/p/3e08e6f6e244

"""
import numpy as np
import pandas as pd


class EntropyMethod:
    def __init__(self, index, positive, negative, row_name):
        if len(index) != len(row_name):
            raise Exception('数据指标行数与行名称数不符')
        if sorted(index.columns) != sorted(positive + negative):
            raise Exception('正项指标加负向指标不等于数据指标的条目数')

        self.index = index.copy().astype('float64')
        self.positive = positive
        self.negative = negative
        self.rowName = row_name
        self.uniformMat = None
        self.pMat = None
        self.entropySeries = None
        self.dSeries = None
        self.Weight = None
        self.score = None
        self.functionSequence()

    def functionSequence(self):
        self.uniform()
        self.calcProbability()
        self.calcEntropy()
        self.calcEntropyRedundancy()
        self.calcWeight()
        self.calcWeight()
        self.calcScore()

    def uniform(self):  # 归一化方法，将传递进来的数据归一化
        uniform_mat = self.index.copy()
        min_index = {column: min(uniform_mat[column]) for column in uniform_mat.columns}
        max_index = {column: max(uniform_mat[column]) for column in uniform_mat.columns}
        for i in range(len(uniform_mat)):
            for column in uniform_mat.columns:
                if column in self.negative:
                    uniform_mat[column][i] = (max_index[column] - uniform_mat[column][i]) / (
                            max_index[column] - min_index[column])
                else:
                    uniform_mat[column][i] = (uniform_mat[column][i] - min_index[column]) / (
                            max_index[column] - min_index[column])

        self.uniformMat = uniform_mat
        return self.uniformMat

    def calcProbability(self):  # 计算指标比重
        p_mat = self.uniformMat.copy()
        for column in p_mat.columns:
            sigma_x_1_n_j = sum(p_mat[column])
            p_mat[column] = p_mat[column].apply(
                lambda x_i_j: x_i_j / sigma_x_1_n_j if x_i_j / sigma_x_1_n_j != 0 else 1e-6)

        self.pMat = p_mat
        return p_mat

    def calcEntropy(self):  # 计算熵值
        e_j = -(1 / np.log(len(self.pMat) + 1)) * np.array(
            [sum([pij * np.log(pij) for pij in self.pMat[column]]) for column in self.pMat.columns])
        ejs = pd.Series(e_j, index=self.pMat.columns, name='指标的熵值')

        self.entropySeries = ejs
        return self.entropySeries

    def calcEntropyRedundancy(self):  # 计算信息熵冗余度
        self.dSeries = 1 - self.entropySeries
        self.dSeries.name = '信息熵冗余度'
        return self.dSeries

    def calcWeight(self):  # 计算权值
        self.uniform()
        self.calcProbability()
        self.calcEntropy()
        self.calcEntropyRedundancy()
        self.Weight = self.dSeries / sum(self.dSeries)
        self.Weight.name = '权值'
        return self.Weight

    def calcScore(self):  # 计算评分
        self.calcWeight()
        self.score = pd.Series(
            [np.dot(np.array(self.index[row:row + 1])[0], np.array(self.Weight)) for row in range(len(self.index))],
            index=self.rowName, name='得分'
        ).sort_values(ascending=False)
        return self.score


if __name__ == '__main__':
    df = pd.read_csv('gdp.csv').dropna().reset_index(drop=True)
    # print(df.head)

    Index = ["GDP总量增速", "人口总量", "人均GDP增速", "地方财政收入总额", "固定资产投资", "社会消费品零售总额增速", "进出口总额",
             "城镇居民人均可支配收入", "农村居民人均可支配收入"]
    Positive = Index
    Negative = []
    Marker = df['地区']
    Index = df[Index]
    eMethod = EntropyMethod(Index, Positive, Negative, Marker)
    print('各项数据权值：\n', eMethod.Weight)
