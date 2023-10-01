# [債務不履行リスクの低減](https://signate.jp/competitions/983)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("債務不履行csv/train.csv")

df_train

# 各列のユニークな値を取得
unique_values = {col: df_train[col].unique() for col in df_train.columns}

print(unique_values)

loan_status_mapping = {"FullyPaid": 0, "ChargedOff": 1}
df_train["loan_status"] = df_train["loan_status"].map(loan_status_mapping)

df_train

y_train = df_train["loan_status"]
x_train = df_train.drop("loan_status", axis = 1)
#目的変数と説明変数の作成




