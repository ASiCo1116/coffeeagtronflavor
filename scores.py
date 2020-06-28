import pandas as pd
import numpy as np

df = pd.read_excel('./data/Chemicals_scores.xlsx')
# print(df)
# print(df['Number'])
# df = df.reset_index().pivot_table(columns=["index"])

df = df.groupby(df['Number'], sort = False).mean()
print(df.iloc[:, -1])
acidity = df.iloc[:, -1].values
print(acidity)