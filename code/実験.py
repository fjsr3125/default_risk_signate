import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  KFold
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import xgboost as xgb
import sweetviz as sv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
# %matplotlib inline 

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

#前処理を行う
#fullypaid = 0, それ以外を1とする
loan_status_mapping = {"FullyPaid": 0, "ChargedOff": 1}
train["loan_status"] = train["loan_status"].map(loan_status_mapping)


# +
# カテゴリカル・データに対してラベルエンコードを行う．
# le = LabelEncoder()
# # train["term"] = le.fit_transform(train["term"])
# train["employment_length"] = le.fit_transform(train["employment_length"])
# # train["purpose"] = le.fit_transform(train["purpose"])
# train["application_type"] = le.fit_transform(train["application_type"])
# # test["term"] = le.fit_transform(test["term"])
# test["employment_length"] = le.fit_transform(test["employment_length"])
# # test["purpose"] = le.fit_transform(test["purpose"])
# test["application_type"] = le.fit_transform(test["purpose"])

# +
#gradeに対して，A1が最も大きく，F5が最も小さくなるようにラベルエンコード
# train["grade"] = train["grade"].astype("category")
# train["grade"] = train["grade"].cat.reorder_categories(train["grade"].cat.categories[::-1])
# train["grade"] = train["grade"].cat.codes
# test["grade"] = test["grade"].astype("category")
# test["grade"] = test["grade"].cat.reorder_categories(test["grade"].cat.categories[::-1])
# test["grade"] = test["grade"].cat.codes
# -

#Holdout Target Encodingの関数定義
def Holdout_te(df,column,target):
  #Kfoldの定義
  kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
  #仮の箱を用意
  box = np.zeros(len(df))
  #Nanで埋めておく
  box[:] = np.nan
  #繰り返しながらTarget Encodingを行う
  for idx1, idx2 in kf.split(df):
    #２分割
    train = df.iloc[idx1]
    val = df[column].iloc[idx2]
    #グループごとの平均を計算
    mean = train.groupby(column)[target].mean()
    #boxにカテゴリ毎の平均を入れる
    for i,m in mean.items():
      for v in val.index:
        if val[v] == i: box[v] = m
  #新たな列に挿入
  df[column + "_target"] = box
  return df


#有用そうなデータに対してターゲットエンコーディングを行う．
train=Holdout_te(train, "term", "loan_status")
train=Holdout_te(train, "grade", "loan_status")
train=Holdout_te(train, "purpose", "loan_status")
train=Holdout_te(train, "employment_length","loan_status")
train=Holdout_te(train, "application_type", "loan_status")

#カテゴリカル変数はモデル作成のために不必要であるので消すから，今のうちにテストデータのターゲットエンコーディング用の値を取っておく
term_target_mean  = train.groupby("term")["term_target"].mean()
grade_target_mean = train.groupby("grade")["grade_target"].mean()
purpose_target_mean = train.groupby("purpose")["purpose_target"].mean()
employment_length_mean = train.groupby("employment_length")["employment_length_target"].mean()
application_type_mean = train.groupby("application_type")["application_type_target"].mean()

#loan_statusに関係のない列を落とす
train = train.drop(["term","grade","purpose","application_type", "employment_length"], axis = 1)

#いらないデータを落とす
train = train.drop(["id"], axis = 1)
test  = test.drop(["id"], axis = 1)

# +
#目的変数と説明変数の作成
y = train["loan_status"]
x = train.drop(["loan_status"], axis = 1)

#訓練用データとバリデーション用データの分類
tr_x, va_x, tr_y, va_y = train_test_split(x, y, test_size = 0.2, random_state=1, stratify=y)
# -

#xgbの型に変換する
dtrain = xgb.DMatrix(tr_x, label = tr_y)

#パラメータの設定
param = {"max_depth": 9, "eta":0.1, "objective": "binary:hinge"}

#学習をしてみる
num_round = 8
bst = xgb.train(param, dtrain, num_round)

dval = xgb.DMatrix(va_x)
pred_train  = bst.predict(dval)

#スコアの確認
score_f1 = f1_score(va_y, pred_train)
print("f1_score:{0:.4f}".format(score_f1))

#テストデータへの適用
test["term_target"] = test["term"].map(term_target_mean)
test["grade_target"] = test["grade"].map(grade_target_mean)
test["purpose_target"] = test["purpose"].map(purpose_target_mean)
test["employment_length"] = test["employment_length"].map(employment_length_mean)
test["application_type"] = test["application_type"].map(application_type_mean)








