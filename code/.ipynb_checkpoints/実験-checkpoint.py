#必要なライブラリのインストール
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# %matplotlib inline 

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

a = train.isnull().count()

a

# +
import seaborn as sns
import matplotlib.pyplot as plt

# データを用意する（例：DataFrameから適切なカラムを抽出）
# ダミーデータの例：
data = sns.load_dataset("iris")

# カラムのリストを作成
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# FacetGridを作成
g = sns.FacetGrid(data, col="species")

# 各カラムにdistplotを適用
g.map(sns.histplot, "sepal_length")
g.map(sns.histplot, "sepal_width")
g.map(sns.histplot, "petal_length")
g.map(sns.histplot, "petal_width")

# グラフを表示
plt.show()


# +
import seaborn as sns
import matplotlib.pyplot as plt

# データを用意する（例：DataFrameから適切なカラムを抽出）
# ダミーデータの例：
data = sns.load_dataset("iris")

# カラムのリストを作成
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# FacetGridを作成
g = sns.FacetGrid(data, col="species")

# 各カラムにdistplotを適用
g.map(sns.histplot, "sepal_length")
g.map(sns.histplot, "sepal_width")
g.map(sns.histplot, "petal_length")
g.map(sns.histplot, "petal_width")

# グラフを表示
plt.show()

# -



# +
import seaborn as sns
import matplotlib.pyplot as plt

# データを用意する（例：DataFrameから適切なカラムを抽出）
# ダミーデータの例：
data = sns.load_dataset("tips")

# カラムのリストを作成
columns = ['total_bill', 'tip', 'size']

# カラムごとに棒グラフを描画
for col in columns:
    sns.barplot(data=data, x="day", y=col)
    plt.show()

# -

dir(train)

vars(train)

train.keys()


