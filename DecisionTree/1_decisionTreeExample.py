from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn import preprocessing
import csv
import graphviz

file = open('AllElectronics.csv', 'r')
reader = csv.reader(file)
header = reader.__next__()  # 第0行
# print(header)

featureList, labelList = [], []
for row in reader.__iter__():  # 这里开始就是第一行了
    labelList.append(row[-1])
    featureDict = {}
    for column in range(1, len(row) - 1):
        featureDict[header[column]] = row[column]
    featureList.append(featureDict)

# print(featureList)

dictVector = DictVectorizer()
xData = dictVector.fit_transform(featureList).toarray()
print('xData:\n', xData)
print('对应xData\n', dictVector.get_feature_names())
# print(labelList)
labelBin = preprocessing.LabelBinarizer()
yLabel = labelBin.fit_transform(labelList)
# print(labelList)
model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(xData, yLabel)
print(model.predict([xData[0]]))  # 要2d 数组

dotData = tree.export_graphviz(model, feature_names=dictVector.get_feature_names(),
                               class_names=labelBin.classes_, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dotData)
graph.render(format='svg')
