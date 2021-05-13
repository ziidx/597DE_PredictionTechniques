import matplotlib.pyplot as plt
import numpy as np

#Plots the performance metrics for linear regression compared to window size
def plot_ideal_linear():
    #Data used for plotting, first index of each error metric array corresponds to wsize 2, second index to wsize 5, etc.
    fig, ax = plt.subplots()
    wsize = [2, 5, 10, 15, 30, 60, 120]
    rmse = [0.5553805407099379, 0.40799314598680503, 0.5462840414172848, 0.6352624006294564, 0.7873806478513212, 0.9359522287368037, 1.1178963371507609]
    mae = [0.12509734151950996, 0.10361423810359142, 0.14837697368762434, 0.18082672614867495, 0.24290734553719653, 0.32024857812330043, 0.45500899332213346]
    mape = [0.0437160472299968, 0.04728591670319416, 0.09315531469856916, 0.1251602757123446, 0.16862225609120948, 0.331724642074517, 0.6700746772343565]

    #plots all 3 error metrics on same figure against window size
    ax.plot(wsize, rmse, label= 'RMSE')
    ax.plot(wsize, mae, label='MAE')
    ax.plot(wsize, mape, label='MAPE')

    #Formatting
    ax.legend()
    plt.xlabel("Window Size")
    plt.ylabel("Error Value")

    plt.show()

#plot_ideal_linear()


#Plots performance comparison for all 3 forecasting techniques
def plot_technique_comparison():
    #Data used for plotting, first value for each array corresponds to RMSE, second value to MAE, etc.
    xlabels = ['RMSE', 'MAE', 'MAPE']
    lr = [0.5553805407099379, 0.12509734151950996, 0.0437160472299968]
    pers = [0.5553796042297725, 0.12509628620552968, 0.04371585909242434]
    arima = [0.5533012075522922, 0.1431868009079184, 0.10892875344765747]

    #used for formatting plot
    x = np.arange(len(xlabels))
    width = 0.2
    
    #Plots the performance metrics for each technique as a bar plot
    plt.bar(x-0.2, lr, width, label="Linear Regression")
    plt.bar(x, pers, width, label="Persistence")
    plt.bar(x+0.2, arima, width, label="ARIMA")

    #plot formatting
    plt.title("Performance Comparison of All 3 Techniques")
    plt.xticks(x, xlabels)
    plt.xlabel("Error Metric")
    plt.ylabel("Error Value")
    plt.legend()


    plt.show()

#plot_technique_comparison()

#Plots performance for each technique over varying resolution input
def plot_var_res():
    fig, ax = plt.subplots()
    resolution = [1, 5, 15, 30, 60]

    #Data for error metrics for persistence when resolution changes, index 0 for each array corresponds to resolution of 1, index 1 -> resolution 5, etc.
    rmse_p = [0.5553796042297725, 1.0394229750216024, 1.3467108576341302, 1.648570084918214, 2.1554820994273216]
    mae_p = [0.3027244976914794, 0.12509628620552968, 0.4907821245224181, 0.6999490712678894, 1.0540133229351385]
    mape_p = [0.04371585909242434, 0.14413329218744503, 0.5147797522923613, 1.021700546582129, 2.489240318293301]

    rmse_lr = [0.5553805407099379, 1.039422990142265, 1.3467108514478716, 1.648570086038203, 2.1554820979714466]
    mae_lr = [0.12509734151950996, 0.3027244696565547, 0.4907821218506787, 0.6999490707406132, 1.0540133223746437]
    mape_lr = [0.0437160472299968, 0.14413328995537267, 0.5147797525402349, 1.0217005379269681, 2.48924031500443]

    rmse_a = [0.5533012075522922, 0.973317314165192, 1.2930000961473112, 1.608770005451177, 2.068558162089965]
    mae_a = [0.1431868009079184, 0.3553421454247975, 0.5785085989238113, 0.8208115915753256, 1.204326227285773]
    mape_a = [0.10892875344765747, 0.33264307332539883, 0.8911421516169342, 1.5636443010296492, 3.0181772414826002]

    #plots all 3 error metrics for chosen technique on same figure against resolution size
    ax.plot(resolution, rmse_a, label= 'RMSE')
    ax.plot(resolution, mae_a, label='MAE')
    ax.plot(resolution, mape_a, label='MAPE')

    #Formatting
    ax.legend()
    plt.xlabel("Resolution")
    plt.ylabel("Error Value")
    plt.title("ARIMA Performance Metrics for varying Resolution")

    plt.show()

plot_var_res()