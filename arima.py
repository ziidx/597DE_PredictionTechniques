import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


df = pd.read_csv('solar_full.csv', index_col=['timestamp']).drop(columns=['UTC'])

pred_dict = {}
#data = df['kW-5'].diff().dropna()
#plot_pacf(data)
#plot_acf(data)
#plt.show()

for x in df:
    model = ARIMA(df[x], order=(1,0,1))
    res = model.fit()
    pred_dict[x] = res.fittedvalues

output = pd.DataFrame.from_dict(pred_dict)
diff = output.subtract(df)
mdf = abs(diff) / df.replace(0,1)

RMSE = ((diff ** 2).mean() ** 0.5).mean()
MAE = abs(diff).mean().mean()
MAPE = mdf.mean().mean()

print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('MAPE: ' + str(MAPE))