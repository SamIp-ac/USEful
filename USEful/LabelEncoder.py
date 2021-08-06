from pyspark.sql import SparkSession
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn import preprocessing
from pyspark.ml.feature import StringIndexer

# LabelBinarizer
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
LabelBinarizer()
lb.classes_
# >>> array([1, 2, 4, 6])
lb.transform([1, 6])
# >>>array([[1, 0, 0, 0],
#          [0, 0, 0, 1]])

# MultiLabelBinarizer
lb = preprocessing.MultiLabelBinarizer()
lb.fit_transform([(1, 2), (3,)])
# >>> array([[1, 1, 0],
#           [0, 0, 1]])
lb.classes_
# >>> array([1, 2, 3])
# LabelEncoder 寫目標對應的class位置, 可用在非數字array
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
le.classes_
# >>> array([1, 2, 6])
le.transform([1, 1, 2, 6])
# >>> array([0, 0, 1, 2])
le.inverse_transform([0, 0, 1, 2])
# >>> array([1, 1, 2, 6])

# StringIndexer

spark = SparkSession.\
            builder.\
            appName('Test').\
            getOrCreate()
df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])

indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
indexed = indexer.fit(df).transform(df)
indexed.show()

# OrdinalEncoder 分類帽
enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
enc.transform([['female', 'from US', 'uses Safari']])
# >>> array([[0., 1., 1.]])
