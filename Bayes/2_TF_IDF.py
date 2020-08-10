"""

新闻分类
    词袋模型
        忽略语法和语序，仅看作是若干个词汇的集合，每一个单词都是独立的
    TF
        提取词频
        出现最多的往往是“的”，“是”等停用词
        而且同频词重要性不一定一样，常见词应该排在后面
    IDF
        逆文档频率
        一个词重要性的权重，大小与一个词的常见程度成反比
        IDF = log(语料库的文档总数 / (包含该词的文档数 + 1))
    TF-IDF = TF x IDF

"""
from sklearn import datasets
from sklearn import model_selection
from sklearn import feature_extraction

news = datasets.fetch_20newsgroups(subset='all')
print(news.target_names)
print(len(news.data))
print(news.data[0])
print('--------------------------------------------------------------------------------------------------------------')
print(news.target[0])
xTrain, xTest, yTrain, yTest = model_selection.train_test_split(news.data, news.target)

vectorizedModel = feature_extraction.text.TfidfVectorizer()
vectorizedModel.fit(news.data)
vector = vectorizedModel.transform([news.data[0]])
print(vector.max())
