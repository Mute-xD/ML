"""

Stacking
    使用多个不同的分类器进行预测，把预测结果作为一个次级分类器的输入，次级输出为最终结果

"""
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier

iris = datasets.load_iris()
xData, yTarget = iris.data[:, 1:3], iris.target

classifier1 = KNeighborsClassifier(n_neighbors=1)
classifier2 = DecisionTreeClassifier()
classifier3 = LogisticRegression()

lowerLayer = LogisticRegression()
stacking = StackingClassifier(classifiers=[classifier1, classifier2, classifier3], meta_classifier=lowerLayer)

for classifier, label in zip([classifier1, classifier2, classifier3, lowerLayer],
                             ['KNN', 'Decision Tree', 'Logistic Regression', 'Stacking Classifier']):
    scores = model_selection.cross_val_score(classifier, xData, yTarget, cv=3, scoring='accuracy')  # cv=3，得到三个结果
    print('Accuracy: %0.2f -> %s' % (scores.mean(), label))
