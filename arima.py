import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv('solar_full.csv', index_col=['timestamp']).drop(columns=['UTC'])
#df = df.iloc[::60] Changes resolution for question 5
#Creates prediction dictionary for storing predicted values for each installation like in linear regression
pred_dict = {}

#Used for plotting acf and pacf for each installation to determine p and q values for ARIMA modelling
#paramer d is given as 0 in the assignment instructions.

#data = df['kW-5'].diff().dropna()
#plot_pacf(data)
#plot_acf(data)
#plt.show()

#Creates ARIMA model and predicts values for each installation, parameter d is given while parameters p and q
#were determined from pacf and acf plots above.
for x in df:
    model = ARIMA(df[x], order=(1,0,1))
    res = model.fit()
    pred_dict[x] = res.fittedvalues

#Similar to linear regression, converts dictionary containing predictions for each installation into a single 
#dataframe for error calculation. See persistence.py for explanation on mdf and MAPE calculation.
output = pd.DataFrame.from_dict(pred_dict)
diff = output.subtract(df)
mdf = abs(diff) / df.replace(0,1)

#Error/Performance Calculation
RMSE = ((diff ** 2).mean() ** 0.5).mean()
MAE = abs(diff).mean().mean()
MAPE = mdf.mean().mean()

print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('MAPE: ' + str(MAPE))