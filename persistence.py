import pandas as pd 
import matplotlib.pyplot as plt



df = pd.read_csv('solar_full.csv', index_col=['timestamp']).drop(columns=['UTC'])
#df = df.iloc[::60] Changes resolution for question 5

#Predictions for persistence is original dataframe shifted down once, however the first row will be all NaNs so that should be ignored
predictions = df.shift().iloc[1:]


#Creates diff dataframe containing differences between predicted value and actual value, compares starting from index 1 because to account for NaN from predictions
diff = predictions.subtract(df.iloc[1:])

#MAPE needs diff/df, but for values in df that are 0, this causes a divide by zero error. In the event that a predicted
#value is a real value but the actual value is 0, the value itself is the error(for instance, predicted value of 0.0052 when
#actual value is 0 means that difference = 0.0052 - 0 = 0.0052, which is an actual value that should be taken into account
#for error calculation. By dividing abs(diff) with a copy of df wherein all 0 values in df are replaced with 1s, we can account for
#nonzero predictions when actual value 0. This also works if predicted value and actual value are 0, so their difference would be 0
#but 0/0 would still cause divide by zero error, but 0/1 = 0. This does not affect calculation where both prediction and df value are nonzero.
#Therefore, we won't be throwing out potential values even if df for a sample is 0.
mdf = abs(diff) / df.replace(0,1)  

#Calculations for each error/performance metric
RMSE = ((diff ** 2).mean() ** 0.5).mean()
MAE = abs(diff).mean().mean()
MAPE = mdf.mean().mean()

print('RMSE: ' + str(RMSE))
print('MAE: ' + str(MAE))
print('MAPE: ' + str(MAPE))

