import pandas as pd 
import matplotlib.pyplot as plt



df = pd.read_csv('solar_small.csv', index_col=['timestamp']).drop(columns=['UTC'])
df2 = df.copy().iloc[1:]
predictions = df.copy().shift().iloc[1:]
pred_dict = {}

diff = predictions.subtract(df2)
mdf = abs(diff) / df2.copy().replace(0,1)

RMSE = ((diff ** 2).mean() ** 0.5).mean()
MAE = abs(diff).mean().mean()
MAPE = mdf.mean().mean()

print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('MAPE: ' + str(MAPE))