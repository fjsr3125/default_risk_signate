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

# カテゴリカル・データに対してラベルエンコードを行う．
le = LabelEncoder()
# train["term"] = le.fit_transform(train["term"])
train["purpose"] = le.fit_transform(train["purpose"])
train["application_type"] = le.fit_transform(train["application_type"])
# test["term"] = le.fit_transform(test["term"])
test["purpose"] = le.fit_transform(test["purpose"])
test["application_type"] = le.fit_transform(test["purpose"])

#termのデータを数字に変換
term_mapping = {"3 years": 3, "5 years": 5}
train["term"] = train["term"].map(term_mapping)
test["term"] = test["term"].map(term_mapping)

# +
#employment_lengthのデータを数字に変換
employment_length_mapping = {"0 years": 0,
                             "1 year": 1,
                             "2 years": 2,
                             "3 years": 3,
                             "4 years": 4,
                             "5 years": 5,
                             "6 years": 6,
                             "7 years": 7,
                             "8 years": 8,
                             "9 years": 9,
                             "10 years": 10,}

train["employment_length"] = train["employment_length"].replace(employment_length_mapping)
# -

#gradeのデータを数字に変換
grade_mapping = {f"{chr(69-i)}{5-j}": i*5+5-j for i in range(5) for j in range(5)}
train["grade"] = train["grade"].map(grade_mapping)
train.info()

train = train.dropna()

train["loan_amnt_log"] = np.log1p(train["loan_amnt"])
train["interest_rate_log"] = np.log1p(train["interest_rate"])
test["loan_amnt_log"] = np.log1p(test["loan_amnt"])
test["interest_rate_log"] = np.log1p(test["interest_rate"])

train

#いらないデータを落とす
train = train.drop(["id", "loan_amnt", "interest_rate"], axis = 1)
test  = test.drop(["id", "loan_amnt", "interest_rate"], axis = 1)

# +
# train["pay_money"] = train["term"] * train["loan_amnt_log"] * train["interest_rate_log"]

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
param = {"max_depth": 8, "eta":0.10, "objective": "binary:hinge"}



#学習をしてみる
num_round = 7
bst = xgb.train(param, dtrain, num_round)

dval = xgb.DMatrix(va_x)
pred_train  = bst.predict(dval)

#スコアの確認
score_f1 = f1_score(va_y, pred_train)
print("f1_score:{0:.4f}".format(score_f1))










