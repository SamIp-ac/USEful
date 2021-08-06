from statsmodels.formula.api import ols
import statsmodels.api as sm
import pandas as pd
import numpy as np
import pyspark
import matplotlib.pyplot as plt


print(pyspark.version)
# Subplot
fig, axes = plt.subplots(2, 2)

data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))

data.plot.bar(ax=axes[1, 1], color='b', alpha=0.5)
data.plot.barh(ax=axes[0, 1], color='k', alpha=0.5)
#

fig, ax1 = plt.subplots(1, 1)  # 做1*1个子图，等价于 " fig, ax1 = plt.subplot() "，等价于 " fig, ax1 = plt.subplots() "

ax2 = ax1.twinx()  # 让2个子图的x轴一样，同时创建副坐标轴。

# 作y=sin(x)函数
x1 = np.linspace(0, 4 * np.pi, 100)
y1 = np.sin(x1)
ax1.plot(x1, y1)

#  作y = cos(x)函数
x2 = np.linspace(0, 4 * np.pi, 100)  # 表示在区间[0, 4π]之间取100个点（作为横坐标，“线段是有无数多个点组成的”）。
y2 = np.cos(x2)

# plot with the respective actual value
x = [1, 2, 3]
y = [9, 8, 7]

plt.plot(x, y)
for a, b in zip(x, y):
    plt.text(a, b, str(b))
plt.show()

# OLS
df = pd.DataFrame()
m = ols('price ~ sqft_living', df).fit()
print(m.summary())

# ColumnTransformer
X = pd.DataFrame()
X_transformer = ColumnTransformer(
    transformers=[
        ("Country",        # Just a name
         OneHotEncoder(),  # The transformer class
         [0]            # The column(s) to be applied on.
         )
    ], remainder='passthrough'
)
X = X_transformer.fit_transform(X)
print(X)

# inner join
table1 = pd.DataFrame()
table2 = pd.DataFrame()
pd.merge(table1, table2, how='inner', on='col name')
# Full Join
pd.merge(table1, table2, how='outer', on='col name')
# Left Join
pd.merge(table1, table2, how='left', on='col name')

# cancel the NaN in column 'col name'
pd.merge(table1, table2, how='outer', on='col name', indicator=True)

# different col name
pd.merge(table1, table2, how='inner', left_on='column name1', right_on='column name2')
# Delete two or more repeated rows
table1 = table1.drop_duplicates()
