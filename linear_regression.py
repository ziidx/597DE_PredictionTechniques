import pandas as pd
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm


df = pd.read_csv('solar_full.csv', index_col=['timestamp']).drop(columns=['UTC'])
wsize = 2

pred_dict = {}
nobs=pd.Series([x for x in range(len(df))], df.index, name='lin_coeff')
lin_coeff = sm.add_constant(nobs)
predictnobs = nobs + 1

for x in df:
    model = RollingOLS(df[x], lin_coeff, window=wsize)
    res = model.fit()
    prediction = pd.Series(res.params['lin_coeff']*predictnobs + res.params['const'])
    pred_dict[x] = prediction

output = pd.DataFrame.from_dict(pred_dict).iloc[wsize-1:]
diff = output.subtract(df.iloc[wsize-1:])
mdf = abs(diff) / df.replace(0,1).iloc[wsize-1:]

RMSE = ((diff ** 2).mean() ** 0.5).mean()
MAE = abs(diff).mean().mean()
MAPE = mdf.mean().mean()

print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('MAPE: ' + str(MAPE))
