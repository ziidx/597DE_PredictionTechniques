import pandas as pd 
import matplotlib.pyplot as plt



df = pd.read_csv('solar_small.csv', index_col=['timestamp'])
df = df.drop(columns=['UTC'])

dfcopy = df.copy().shift()
df = df.drop(index=['2/2/2015 15:25'])
dfcopy = dfcopy.drop(index=['2/2/2015 15:25'])
differences = dfcopy.subtract(df)
mape_df = abs(differences) / df
#mape_df = mape_df.add(0, fill_value = 0)
RMSE = ((differences ** 2).mean() * 0.5).mean()
MAE = abs(differences).mean().mean()
MAPE = mape_df.mean()


print(MAPE)

#df.plot()
#plt.show()