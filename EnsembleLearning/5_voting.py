"""

Voting
    与Stacking类似
    但无次级分类器

"""
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

iris = datasets.load_iris()
xData, yTarget = iris.data[:, 1:3], iris.target

classifier1 = KNeighborsClassifier(n_neighbors=1)
classifier2 = DecisionTreeClassifier()
classifier3 = LogisticRegression()

voting = VotingClassifier([('KNN', classifier1), ('DecisionTree', classifier2), ('LogisticRegression', classifier3)])
for classifier, label in zip([classifier1, classifier2, classifier3, voting],
                             ['KNN', 'Decision Tree', 'Logistic Regression', 'Voting']):
    scores = model_selection.cross_val_score(classifier, xData, yTarget, cv=3, scoring='accuracy')
    print('Accuracy: %0.2f -> %s' % (scores.mean(), label))
