# [【第39回_Beginner限定コンペ】債務不履行リスクの低減](https://signate.jp/competitions/983#abstract)

#必要なライブラリのインストール
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  KFold
from sklearn.metrics import log_loss
import xgboost as xgb
import sweetviz as sv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore")
# %matplotlib inline 

# # EDAをおこなう

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# +
#パラメータの設定
#クロスバリデーションの数を設定．
nr_cv = 5

#目的変数の設定
target = "loan_status"

#以下のしきい値を超えたColumnを用いる．
min_val_corr = 0.4

# +

print(train.shape)
print("*" * 50)
print(test.shape)
print("*" * 50) 
print(train.info())
print("*" * 50)
print(test.info())
# -

train.head()

test.head()

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

train.describe()

test.describe()

numerical_feats = ["id", "loan_amnt", "interest_rate", "credit_score"]
for col in numerical_feats:
    print('{:15}'.format(col), 
          'Mean: {:05.2f}'.format(train[col].mean()) , 
          '   ' ,
          'Std: {:05.2f}'.format(train[col].std()) , 
          '   ' ,
          'Skewness: {:05.2f}'.format(train[col].skew()) , 
          '   ' ,
          'Kurtosis: {:06.2f}'.format(train[col].kurt())  
         )

# +
numerical_feats = train.dtypes[train.dtypes != "object"].index
print("Number of Numerical features: ", len(numerical_feats))

categorical_feats = train.dtypes[train.dtypes == "object"].index
print("Number of Categorical features: ", len(categorical_feats))
# -

column = train.columns
for col in column:
    print(f"{col}の特有の値は，{train[col].unique()}" )

loan_status_mapping = {"FullyPaid": 0, "ChargedOff": 1}
train["loan_status"] = train["loan_status"].map(loan_status_mapping)

train["loan_status"] = train["loan_status"].astype(float)

test

#loan_statusに関係のない列を落とす
train = train.drop(["id"], axis = 1)
test  = test.drop(["id"], axis = 1)

train["grade"].unique().sort()
train["grade"].sort_values().unique()

le = LabelEncoder()
train["term"] = le.fit_transform(train["term"])
train["grade"] = le.fit_transform(train["grade"])
train["employment_length"] = le.fit_transform(train["employment_length"])
train["purpose"] = le.fit_transform(train["purpose"])
train["application_type"] = le.fit_transform(train["application_type"])
test["term"] = le.fit_transform(test["term"])
test["grade"] = le.fit_transform(test["grade"])
test["employment_length"] = le.fit_transform(test["employment_length"])
test["purpose"] = le.fit_transform(test["purpose"])
test["application_type"] = le.fit_transform(test["purpose"])

train["loan_amnt_log"] = np.log1p(train["loan_amnt"])
train["credit_score_log"] = np.log10(train["credit_score"])
train["intrest_rate_log"] = np.log1p(train["interest_rate"])
test["loan_amnt_log"] = np.log1p(test["loan_amnt"])
test["credit_score_log"] = np.log1p(test["credit_score"])
test["intrest_rate_log"] = np.log1p(test["interest_rate"])

train

# ## 
# それぞれの列について調べていく

# #### loan_statusについて

sns.countplot(x = "loan_status", data = train)

# ### loan_amntについて

sns.displot(x = train["loan_amnt"], hue = train["loan_status"])

sns.displot(x = train["loan_amnt_log"])

sns.countplot(x = "term", hue = "loan_status", data = train)
#3年のほうがFullypaidの割合が高くなっている．

# ### gradeについて

sns.countplot(x = "grade", hue = "loan_status", data = train)

sns.displot(x = train["credit_score_log"])

sns.displot(x = train["intrest_rate_log"])

#それぞれのデータごとに比率を計算する
category_rations_grade = train.groupby("grade")["loan_status"].mean()
category_rations_grade

# ### interest_rateについて

sns.displot(x = "interest_rate", hue = "loan_status", data = train, kind = "kde")
#利率が高いと，返済できづらくなっている．

# ### purposeについて

sns.displot(x = "purpose", data = train, hue = "loan_status", aspect = 2)
#比率としてはあまり変わらなさそう

category_rations_purpose = train.groupby("purpose")["loan_status"].mean()
category_rations_purpose

# ### credit_scoreについて

sns.displot(x = "credit_score", hue = "loan_status", kind = "kde", data = train)
#クレジットスコアが高い人は，完済している．
#750を超えると，ほぼいない，

# ### application_typeについて

sns.displot(x = "application_type", hue = "loan_status", data = train)

#それぞれのデータごとに比率を計算する
category_rations_app = train.groupby("application_type")["loan_status"].mean()
category_rations_app
#ほとんど変わらなそう

# ### employment_lengthについて

sns.displot(x = "employment_length", hue = "loan_status", data = train)

#それぞれのデータごとに比率を計算する
category_rations_emp = train.groupby("employment_length")["loan_status"].mean()
category_rations_emp
#ほとんど変わらなそう



# ### 学習用データを作成

# +
#目的変数と説明変数の作成
y = train["loan_status"]
x = train.drop("loan_status", axis = 1)

#訓練用データとバリデーション用データの分類
tr_x, va_x, tr_y, va_y = train_test_split(x, y, test_size = 0.2, random_state=1, stratify=y)
# -

print(x)

# ### モデルの作成 (xgboostで行う)

# +
dtrain = xgb.DMatrix(tr_x, label = tr_y)
dvalid = xgb.DMatrix(va_x, label = va_y)
dtest = xgb.DMatrix(test)

#ハイパーパラメータの設定
params = {"objective": "binary:logistic", "verbosity":1, "random_state": 71}
num_round = 50

#学習の実行
watchlist = [(dtrain, "train"), (dvalid, "eval")]
model = xgb.train(params, dtrain, num_round, evals = watchlist)

#バリデーションデータでのスコアの確認
va_pred = model.predict(dvalid)
score = log_loss(va_y, va_pred)
print(f"logloss:{score:.4f}")

pred = model.predict(dtest)

# -

pred_value = [0] * len(pred)

for i in range(len(pred)):
    if pred[i] > 0.5:
        pred_value[i] = 1
    else:
        pred_value[i] = 0

pred_value = np.array(pred_value)

# +
y_pred_judge_df = pd.DataFrame(pred_value)

#idを作成
id = np.arange(242150,269050)

#idをDataFrameに変換
id_df = pd.DataFrame(id)

id_df

#id_df と y_pred_judge_df の結合
df_submit = pd.concat([id_df, y_pred_judge_df], axis = 1) 

df_submit

#df_submitの1列目をint型に変更
df_submit = df_submit.astype(int)
#df_submitをcsvファイルに変換
df_submit.to_csv("3rdsubmit_default.csv", index = None, header = None)
# -






