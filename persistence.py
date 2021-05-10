import pandas as pd 
import matplotlib.pyplot as plt



df = pd.read_csv('solar_small.csv', index_col=['timestamp'])
df = df.drop(columns=['UTC'])

dfcopy = df.copy().shift()
df = df.iloc[1:]
dfcopy = dfcopy.iloc[1:]
differences = dfcopy.subtract(df)
mape_df = abs(differences) / df
RMSE = ((differences ** 2).mean() ** 0.5).mean()
MAE = abs(differences).mean().mean()
MAPE = mape_df.mean()
print(RMSE)