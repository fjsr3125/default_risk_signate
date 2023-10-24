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

# ## EDAを行う

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

#前処理を行う
#fullypaid = 0, それ以外を1とする
loan_status_mapping = {"FullyPaid": 0, "ChargedOff": 1}
train["loan_status"] = train["loan_status"].map(loan_status_mapping)

# +
#sweetvisを使う
# report = sv.analyze(train, target_feat = "loan_status")

# +
#結果の表示
# report.show_html("eda_default_risk.html")
# -

#カテゴリカル・データに対してラベルエンコードを行う．
le = LabelEncoder()
train["purpose"] = le.fit_transform(train["purpose"])
train["application_type"] = le.fit_transform(train["application_type"])
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
test["employment_length"] = test["employment_length"].replace(employment_length_mapping)
# -

#gradeのデータを数字に変換
grade_mapping = {f"{chr(69-i)}{5-j}": i*5+5-j for i in range(5) for j in range(5)}
train["grade"] = train["grade"].map(grade_mapping)
test["grade"] = test["grade"].map(grade_mapping)

#偏っているデータに対して対数変換を行う
train["loan_amnt_log"] = np.log1p(train["loan_amnt"])
train["interest_rate_log"] = np.log1p(train["interest_rate"])
test["loan_amnt_log"] = np.log1p(test["loan_amnt"])
test["interest_rate_log"] = np.log1p(test["interest_rate"])

#新しい特徴量を作成する
train["credit_grade_score"] = train["credit_score"] * train["grade"]
train["grade_interest_score"] = train["grade"] / train["interest_rate"]
train["credit_interest_score"] = train["credit_score"] / train["interest_rate"]
test["credit_grade_score"] = test["credit_score"] * test["grade"]
test["grade_interest_score"] = test["grade"] / test["interest_rate"]
test["credit_interest_score"] = test["credit_score"] / test["interest_rate"]

train.columns.unique()

#いらないデータを落とす
train_change = train.drop(["id", "term", "loan_amnt", "interest_rate", "grade", "credit_score", "interest_rate_log"], axis = 1)
test_change = test.drop(["id", "term", "loan_amnt", "interest_rate", "grade", "credit_score", "interest_rate_log"], axis = 1)

train_change

# +
# # sweetvisを使う
# report = sv.analyze(train_change, target_feat = "loan_status")
# #結果の表示
# report.show_html("eda_default_risk_change.html")
# -

# ## とりあえず前処理とかせずにxgboostに入れてみる．

# +
#目的変数と説明変数の作成
y = train["loan_status"]
x = train.drop(["loan_status"], axis = 1)

#訓練用データとバリデーション用データの分類
tr_x, va_x, tr_y, va_y = train_test_split(x, y, test_size = 0.2, random_state=1, stratify=y)
# -

#xgbの型に変換する
dtrain = xgb.DMatrix(tr_x, label = tr_y)
dval = xgb.DMatrix(va_x)

#パラメータの設定
param = {"max_depth": 8, "eta":0.1, "objective": "binary:hinge", "eval_metric": "logloss"}
watchlist = [(dtrain, "train"), (dval, "eval")]

#学習をしてみる
num_round = 8
bst = xgb.train(param, dtrain, num_round)

dval = xgb.DMatrix(va_x)
pred_train  = bst.predict(dval)

#スコアの確認
score_f1 = f1_score(va_y, pred_train)
print("f1_score:{0:.4f}".format(score_f1))

dtest = xgb.DMatrix(test)
pred_test = bst.predict(dtest)

# +
#一応提出してみる
pred_test_df = pd.DataFrame(pred_test)

#idを作成
id = np.arange(242150,269050)

#idをDataFrameに変換
id_df = pd.DataFrame(id)

#id_df と y_pred_judge_df の結合
df_submit = pd.concat([id_df, pred_test_df], axis = 1) 

df_submit

#df_submitの1列目をint型に変更
df_submit = df_submit.astype(int)
#df_submitをcsvファイルに変換
df_submit.to_csv("../submission/4th_submit_default.csv", index = None, header = None)
# -






















