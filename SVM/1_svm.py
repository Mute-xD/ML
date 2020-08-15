"""

SVM
    SVM寻找区分两类的超平面,使边际最大
    边际函数：wx + b = (+-)1
    w, x 为矩阵
    离分界线最近，分别垂直于决策边界的数据点，即为支持向量
    

"""
from sklearn import svm
xData = [[3, 3], [4, 3], [1, 1]]
yLabel = [1, 1, -1]

svmModel = svm.SVC(kernel='linear')
svmModel.fit(xData, yLabel)
print('Support vectors\n', svmModel.support_vectors_)  # 支持向量
print('Which data is support vector ->', svmModel.support_)
print('How much support vector on each sides ->', svmModel.n_support_)
print(svmModel.predict([[5, 5]]))
# wx + b = 0
print('w:', svmModel.coef_)
print('b:', svmModel.intercept_)
