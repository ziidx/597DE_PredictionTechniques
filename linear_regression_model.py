import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('solar_small.csv')
df = df.drop(columns=['UTC'])

# adding another column to iterate through
df['Ticks'] = range(0, len(df.index.values))

X = sm.add_constant(df['Ticks'])
model = sm.OLS(df['kW-2'], X)
results = model.fit()

plt.scatter(df['Ticks'], df['kW-2'], alpha=0.3)
y_predict = results.params[0] + results.params[1]*df['Ticks']
plt.plot(df['Ticks'], y_predict, linewidth=3)
plt.show()

print(results.summary())

