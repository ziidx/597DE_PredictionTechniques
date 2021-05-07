import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import matplotlib



location = r'C:\Users\gchia\Desktop\597DE\solar_201502_201512_clean.csv'
df = pd.read_csv(location, parse_dates=True, index_col=1)
df = df.drop(columns=['UTC'])
dfcopy = df.shift()
dfcopy = dfcopy.drop([0])
print(dfcopy)


#for index in df:
    #if index != 'UTC':
        #print(df[index])
#df.plot()
#plt.show()