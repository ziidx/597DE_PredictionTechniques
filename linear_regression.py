import pandas as pd
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm


df = pd.read_csv('solar_full.csv', index_col=['timestamp']).drop(columns=['UTC'])
#df = df.iloc[::60] Changes resolution for question 5

#Creates dictionary for holding predicted values for each installation that will later be converted to a dataframe
#nobs represents the number of observations/samples from the dataframe and lin_coeff adds an intercept column for 
#rolling regression predictions. The predictnobs is the incremented version of nobs which is used for
#predicting the value at t+1 using coefficients generated at t using rolling regression.
wsize = 2
pred_dict = {}
nobs=pd.Series([x for x in range(len(df))], df.index, name='lin_coeff')
lin_coeff = sm.add_constant(nobs)
predictnobs = nobs + 1

#iterates through each installation, performs linear regression, and stores the predicted values into pred_dict
for x in df:
    model = RollingOLS(df[x], lin_coeff, window=wsize)
    res = model.fit()
    prediction = pd.Series(res.params['lin_coeff']*predictnobs + res.params['const'])
    pred_dict[x] = prediction

#creates dataframe for all the predicted values for all 10 installations. Dataframe data starts from wsize-1 index
#because the linear regression model cannot predict the first few values less than the specified window size.
#See persistence.py for explanation on mdf and MAPE calculation
output = pd.DataFrame.from_dict(pred_dict).iloc[wsize-1:]
diff = output.subtract(df.iloc[wsize-1:])
mdf = abs(diff) / df.replace(0,1).iloc[wsize-1:]

#Calculations for each error/performance metric
RMSE = ((diff ** 2).mean() ** 0.5).mean()
MAE = abs(diff).mean().mean()
MAPE = mdf.mean().mean()

print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('MAPE: ' + str(MAPE))
