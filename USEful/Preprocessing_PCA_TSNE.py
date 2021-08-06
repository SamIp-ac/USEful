from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns

scaler = StandardScaler()

df = pd.read_csv(r'../iot_telemetry_data.csv', skip_blank_lines=True)
df = pd.DataFrame(df)
df = df.dropna()

label_map = {'b8:27:eb:bf:9d:51': 1, '00:0f:00:70:91:0a': 2, '1c:bf:ce:15:ec:4d': 3}
df['device'] = df['device'].map(label_map)

data = df.drop(labels=['ts', 'light', 'motion'], axis=1)
features = ['co', 'humidity', 'lpg', 'smoke', 'temp']

M = df[features]
correlated = M.corr()
print(correlated)
sns.heatmap(correlated)
plt.show()

# PCA
pca = PCA(n_components=2)

X = df[features].values
y = df['device'].values
X = scaler.fit_transform(X)

PrincipalComponents = pca.fit_transform(X)
print('The variance ratio is : ', pca.explained_variance_ratio_)
print('The variance is : ', pca.explained_variance_)
plt.figure(figsize=(8, 6))
plt.scatter(PrincipalComponents[:, 0], PrincipalComponents[:, 1], s=1,  c=y, alpha=1
            , cmap=plt.cm.get_cmap('nipy_spectral', 3))
plt.colorbar()
plt.show()

pca_df = pd.DataFrame(data=PrincipalComponents, columns=['PC1', 'PC2'])
print(pca_df)

# pca can put in new data to do the same data operation by using pca.transform(new_data)
# but T-SNE cannot

# TS_NE = TSNE(n_components=2)
# PrincipalComponents = TS_NE.fit_transform(X)
# print('The variance ratio is : ', pca.explained_variance_ratio_)
# print('The variance is : ', pca.explained_variance_)
# plt.figure(figsize=(8, 6))
# plt.scatter(PrincipalComponents[:, 0], PrincipalComponents[:, 1]
#             , c=y, alpha=15, cmap=plt.cm.get_cmap('nipy_spectral', 10))
# plt.colorbar()
# plt.show()


# pr = PolynomialFeatures(degree=2, include_bias=False)
# x_polly = pr.fit_transform(X_train[['Open', 'High']])