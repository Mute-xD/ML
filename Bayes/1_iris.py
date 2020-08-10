"""

贝叶斯分析
    主要和文本相关
    多项式模型：重复出现的词会进行重复计算
    伯努利模型：重复出现的模型只视其出现一次
    混合模型：训练时考虑，测试时不考虑
    高斯模型：将连续变量转为离散的值（区间分析），适合连续变量

"""
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import naive_bayes

iris = datasets.load_iris()
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(iris.data, iris.target)

multiModel = naive_bayes.MultinomialNB()
multiModel.fit(xTrain, yTrain)
print('Multinomial:\n', metrics.classification_report(yTest, multiModel.predict(xTest)))

bernoulliModel = naive_bayes.BernoulliNB()
bernoulliModel.fit(xTrain, yTrain)
print('Bernoulli:\n', metrics.classification_report(yTest, bernoulliModel.predict(xTest)))

gaussianModel = naive_bayes.GaussianNB()
gaussianModel.fit(xTrain, yTrain)
print('Gaussian:\n', metrics.classification_report(yTest, gaussianModel.predict(xTest)))
