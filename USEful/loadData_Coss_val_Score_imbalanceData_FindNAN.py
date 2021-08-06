from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from keras.utils.data_utils import get_file

f = np.load(get_file("mnist.npz", origin="https://s3.amazonaws.com/img-datasets/mnist.npz"))

f0 = np.load(get_file("mnist.npz", origin="~/.keras"))
x_train = f0['x_train']
y_train = f0['y_train']
x_test = f0['x_test']
y_test = f0['y_test']
f0.close()

cv = cross_val_score(model, X, y, cv=5, scoring='accuracy')
score = np.mean(cv)
print(score)

# choose n_neighbors
k_range = range(1, 31)
k_scores = []

for k_number in k_range:
    knn = KNeighborsClassifier(n_neighbors=k_number)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
#
# 70% train, 15% val, 15% test
# 80% train, 10% val, 10% test
# 60% train, 20% val, 20% test
###

from sklearn import metrics

print('accuracy', metrics.accuracy_score(y_test, predicted))
print('f1 score macro', metrics.f1_score(y_test, predicted, average='macro'))
print('f1 score micro', metrics.f1_score(y_test, predicted, average='micro'))
print('precision score', metrics.precision_score(y_test, predicted, average='macro'))
print('recall score', metrics.recall_score(y_test, predicted, average='macro'))
print('hamming_loss', metrics.hamming_loss(y_test, predicted))
print('classification_report', metrics.classification_report(y_test, predicted))
print('jaccard_similarity_score', metrics.jaccard_score(y_test, predicted))
print('log_loss', metrics.log_loss(y_test, predicted))
print('zero_one_loss', metrics.zero_one_loss(y_test, predicted))
print('AUC&ROC', metrics.roc_auc_score(y_test, predicted))
print('matthews_corrcoef', metrics.matthews_corrcoef(y_test, predicted))

# imbalance data
from matplotlib import pyplot as plt
import seaborn as sns
# To balance the dataset
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import metrics
# Count how many different
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

counter = Counter(y)
print(counter)
# data imbalance
over = SMOTE()
X, y = over.fit_resample(X, y)

# Evaluation Metrics
print(f"Precision score: {precision_score(y_test, y_preds5)}")
print(f"Recall Score : { recall_score(y_test, y_preds5)}")
print(f"F1 Score : {f1_score(y_test, y_preds5)}")
print()
print("-------------Classification Report_________")
print(classification_report(y_test, y_preds5))
print()
sns.heatmap(confusion_matrix(y_test, y_preds5), annot=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
# Find NAN
import pandas as pd
import numpy as np
# import data
df = pd.read_csv(r'../heart.csv')
df = np.array(df)
print('The empty entries of X is : ', len(np.where(np.isnan(df))[0]))
