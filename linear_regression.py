import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import numpy as np 


df = pd.read_csv('solar_201502_201512_clean.csv', index_col=['timestamp'])
df = df.drop(columns=['UTC'])
nobs=pd.Series([x for x in range(len(df))], df.index, name='lin_coeff')
lin_coeff = sm.add_constant(nobs)
predictnobs = pd.Series([x for x in range(1,len(df)+1)], df.index)
installation_mean = pd.Series(df.mean(axis=1), df.index)
wsize = 2


model = RollingOLS(installation_mean, lin_coeff, window=wsize)
res = model.fit()
predictions = pd.Series(res.params['lin_coeff']*predictnobs + res.params['const'])
output = pd.DataFrame({'predictions':predictions, 'mean_10_installations': installation_mean}).iloc[wsize-1:]
differences = output['predictions'] - output['mean_10_installations']
RMSE = ((differences ** 2).mean())**0.5
print(RMSE)

