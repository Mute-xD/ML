"""

SVM 线性分类实例

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

xData = np.r_[np.random.randn(200, 2) - [3, 3], np.random.randn(200, 2) + [3, 3]]
yData = [0] * 200 + [1] * 200
svmModel = svm.SVC(kernel='linear')
svmModel.fit(xData, yData)

print('w:', svmModel.coef_)
print('b:', svmModel.intercept_)

# Plot surface x1w1 + x2w2 + b = 0
plt.scatter(xData[:, 0], xData[:, 1], c=yData, cmap='tab10')
x = np.array([-5, 0, 5])


k = - svmModel.coef_[0][0] / svmModel.coef_[0][1]
b = - svmModel.intercept_ / svmModel.coef_[0][1]
y = k * x + b
plt.plot(x, y, c='k')
lowerVector = svmModel.support_vectors_[0]
upperVector = svmModel.support_vectors_[1]
yLower = k * x + (lowerVector[1] - k * lowerVector[0])
yUpper = k * x + (upperVector[1] - k * upperVector[0])
plt.plot(x, yUpper, 'r--')
plt.plot(x, yLower, 'b--')
plt.show()
