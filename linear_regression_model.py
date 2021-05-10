
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()

df = pd.read_csv('solar_small.csv', parse_dates=['timestamp'])
df = df.drop(columns=['UTC', 'timestamp'])
df['mean'] = df.mean(axis=1)

# adding another column to iterate through
df['Ticks'] = range(0, len(df.index.values))
print(df.head())


# function to locate column
def locate_column(dataframe, column):
    return dataframe.loc[:, column]


# create prediction array
predicted = []


# linear regression algorithm
def linear_regression(window_size):
    endog = locate_column(df, 'mean')
    exog = sm.add_constant(locate_column(df, 'Ticks'))

    rols = RollingOLS(endog, exog, window=window_size)
    rres = rols.fit()

    for i in range(len(df['mean'])):
        m = rres.params['Ticks'][i]
        b = rres.params['const'][i]
        predicted.append(m * (i+1) + b)


# run data with different window sizes
linear_regression(window_size=2)
# linear_regression(window_size=5)
# linear_regression(window_size=10)
# linear_regression(window_size=15)
# linear_regression(window_size=30)
# linear_regression(window_size=60)
# linear_regression(window_size=120)

df['predicted'] = predicted

print(df)

plt.plot(predicted)
plt.plot(df['Ticks'], df['mean'])
plt.show()





