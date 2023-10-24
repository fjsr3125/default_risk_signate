#ライブラリをダウンロード
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  KFold
import sweetviz as sv
pd.set_option('display.max_rows', 10)

df = pd.read_csv("債務不履行csv/train.csv")

loan_status_mapping = {"FullyPaid": 0, "ChargedOff": 1}
df["loan_status"] = df["loan_status"].map(loan_status_mapping)

df["loan_status"] = df["loan_status"].astype(float)

#loan_statusに関係のない列を落とす
df = df.drop(["id","employment_length","application_type"], axis = 1)

df


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



df=Holdout_te(df, "term", "loan_status")
df=Holdout_te(df, "grade", "loan_status")
df=Holdout_te(df, "purpose", "loan_status")

df

#カテゴリカル変数はモデル作成のために不必要であるので消すから，今のうちにテストデータのターゲットエンコーディング用の値を取っておく
term_target_mean  = df.groupby("term")["term_target"].mean()
term_target_mean
grade_target_mean = df.groupby("grade")["grade_target"].mean()
grade_target_mean
purpose_target_mean = df.groupby("purpose")["purpose_target"].mean()
purpose_target_mean

#loan_statusに関係のない列を落とす
df = df.drop(["term","grade","purpose"], axis = 1)

# +
#目的変数と説明変数の作成
y = df["loan_status"]
x = df.drop("loan_status", axis = 1)

#訓練用データを標準化してdataframeに直す
sc = StandardScaler()
x_std = sc.fit_transform(x)
x_std = pd.DataFrame(x_std)
x_std.columns = x.columns
x_std

# +
#訓練用データとテスト用データの分類
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = 0.2, random_state=1, stratify=y)

#ロジスティック回帰モデルの作成
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1, random_state=1, solver="lbfgs", multi_class="ovr")
lr.fit(x_train, y_train)

# +
#ロジスティック回帰モデルから予測値を出す
y_pred = lr.predict(x_test)

#予測と実際のラベルを比較し，性能指標を計算する
from sklearn.metrics import accuracy_score, f1_score

#正解率を計算
accuracy = accuracy_score(y_test, y_pred)
print("正解率：", accuracy)
#f値を計算
f_score = f1_score(y_test, y_pred)
print("F値", f_score)

# +
unique_values, counts = np.unique(y_pred, return_counts=True)

# 結果を表示
for value, count in zip(unique_values, counts):
    print(f"値: {value}, 出現回数: {count}")
# -



df_judge = pd.read_csv("債務不履行csv/test.csv")

#loan_statusに関係のない列を落とす
df_judge = df_judge.drop(["id","employment_length","application_type"], axis = 1)

#テストデータへの適用
df_judge["term_target"] = df_judge["term"].map(term_target_mean)
df_judge["grade_target"] = df_judge["grade"].map(grade_target_mean)
df_judge["purpose_target"] = df_judge["purpose"].map(purpose_target_mean) 

#loan_statusに関係のない列を落とす(説明変数の作成)
x_test = df_judge.drop(["term","grade","purpose"], axis = 1)

x_test

#testデータを標準化してdataframeに直す
x_test_std = sc.fit_transform(x_test)
x_test_std = pd.DataFrame(x_test_std)
x_test_std.columns = x_test.columns

mean_purpose_target = x_test_std["purpose_target"].mean()
x_test_std["purpose_target"].fillna(mean_purpose_target, inplace = True)

#df_judge_stdから予測する
y_pred_judge = lr.predict(x_test_std)

# +
unique_values, counts = np.unique(y_pred_judge, return_counts=True)

# 結果を表示
for value, count in zip(unique_values, counts):
    print(f"値: {value}, 出現回数: {count}")

# +
#y_pred_judgeをDataFrameに変換
y_pred_judge_df = pd.DataFrame(y_pred_judge)

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
df_submit.to_csv("2ndsubmit_default.csv", index = None, header = None)
# -




