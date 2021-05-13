import pandas as pd 
import matplotlib.pyplot as plt



df = pd.read_csv('solar_full.csv', index_col=['timestamp']).drop(columns=['UTC'])

predictions = df.shift().iloc[1:]
pred_dict = {}

diff = predictions.subtract(df.iloc[1:])
mdf = abs(diff) / df.replace(0,1)

RMSE = ((diff ** 2).mean() ** 0.5).mean()
MAE = abs(diff).mean().mean()
MAPE = mdf.mean().mean()

print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('MAPE: ' + str(MAPE))