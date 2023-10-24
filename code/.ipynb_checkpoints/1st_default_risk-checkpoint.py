# [債務不履行リスクの低減](https://signate.jp/competitions/983)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 10)

df_train = pd.read_csv("債務不履行csv/train.csv")

#id列で学習しては良くないのでid列の削除
df_train = df_train.drop(["id"], axis=1)

# +
#loan_statusのデータを数字に変換
loan_status_mapping = {"FullyPaid": 0, "ChargedOff": 1}
df_train["loan_status"] = df_train["loan_status"].map(loan_status_mapping)

#termのデータを数字に変換
term_mapping = {"3 years": 3, "5 years": 5}
df_train["term"] = df_train["term"].map(term_mapping)
df_train

#gradeのデータを数字に変換
grade_mapping = {f"{chr(69-i)}{5-j}": i*5+j+1 for i in range(5) for j in range(5)}
df_train["grade"] = df_train["grade"].map(grade_mapping)
df_train

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

df_train["employment_length"] = df_train["employment_length"].replace(employment_length_mapping)

#purposeのデータを数字に変換(onehotエンコーディングを用いる)
df_train = pd.get_dummies(df_train, columns = ["purpose"], dtype = int)

#application_typeのデータを数字に変換
df_train = pd.get_dummies(df_train, columns = ["application_type"], dtype = int)
# -

df_train.columns

#NaN値の入っている行の削除
df_train.dropna(inplace=True)

# +
#目的変数と説明変数の作成
y = df_train["loan_status"]
x = df_train.drop("loan_status", axis = 1)

#訓練用データを標準化して
sc = StandardScaler()
x_std = sc.fit_transform(x)
# -

#dataframeに直す
x_std = pd.DataFrame(x_std)


#カラム名の割当
x_std.columns = x.columns

# +
#訓練用データとテスト用データの分類
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = 0.3, random_state=1, stratify=y)

#ロジスティック回帰モデルの作成
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100, random_state=1, solver="lbfgs", multi_class="ovr")
lr.fit(x_train, y_train)

# +
y_pred = lr.predict(x_test)

#予測と実際のラベルを比較し，性能指標を計算する
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#正解率を計算
accuracy = accuracy_score(y_test, y_pred)
print("正解率：", accuracy)

# +
unique_values, counts = np.unique(y_pred, return_counts=True)

# 結果を表示
for value, count in zip(unique_values, counts):
    print(f"値: {value}, 出現回数: {count}")
# -



# ["判別用データ"]

#判別用データの作成
df_judge = pd.read_csv("債務不履行csv/test.csv")

#id列を反映しては良くないのでid列の削除
df_judge = df_judge.drop(["id"], axis=1)

# +
#df_judgeのデータを変換

#termのデータを数字に変換
term_mapping = {"3 years": 3, "5 years": 5}
df_judge["term"] = df_judge["term"].map(term_mapping)

#gradeのデータを数字に変換
grade_mapping = {f"{chr(69-i)}{5-j}": i*5+j+1 for i in range(5) for j in range(5)}
df_judge["grade"] = df_judge["grade"].map(grade_mapping)

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

df_judge["employment_length"] = df_judge["employment_length"].replace(employment_length_mapping)

#purposeのデータを数字に変換(onehotエンコーディングを用いる)
df_judge= pd.get_dummies(df_judge, columns = ["purpose"], dtype = int)

#application_typeのデータを数字に変換
df_judge = pd.get_dummies(df_judge, columns = ["application_type"], dtype = int)

# +
#訓練用データを標準化して，dataframeに直す
sc = StandardScaler()
df_judge_std = sc.fit_transform(df_judge)
df_judge_std = pd.DataFrame(df_judge_std)
df_judge_std

#カラム名を割り当てる
df_judge_std.columns = df_judge.columns
# -

#学習用データになかったpurpose_movingというColumnを削除する
df_judge_std.drop(columns = ["purpose_moving"], inplace=True)

#df_judge_stdから予測する
y_pred_judge = lr.predict(df_judge_std)
y_pred_judge.sum()

# +
unique_values, counts = np.unique(y_pred_judge, return_counts=True)

# 結果を表示
for value, count in zip(unique_values, counts):
    print(f"値: {value}, 出現回数: {count}")
# -

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
df_submit.iloc[:,0].astype(int)

#df_submitをcsvファイルに変換
df_submit.to_csv("submit_default.csv", index = None, header = None)






