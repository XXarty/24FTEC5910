import requests
import datetime
import pandas as pd
import json
import time
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import quantstats as qs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import openpyxl
warnings.filterwarnings("ignore")

output_path = 'C://Users//garet//OneDrive//文件//Crypto Trading//github//Crypto-Trading//output'

def generate_metrics(df, title = '', periods = 365):
    test = df.copy()
    test['date'] = pd.to_datetime(test['date'])
    test['date'] = test['date'].dt.tz_localize(None) 
    test.set_index('date', inplace=True)
    # test.index = test.index.tz_localize(None)
    DAYS_PER_YEAR = 365
    print(f'\033[92m\n\033[01m{"PERFORMANCE METRICS" if not title else title}\033[0m',
          f'Cumulative Return:\t{round(((test["return"].add(1).cumprod().sub(1)).iloc[-1])*100,2)}%',
          f'Sharpe:\t\t\t{round(qs.stats.sharpe(test["return"], periods=periods), 2)}',
          f'CAGR:\t\t\t{round(qs.stats.cagr(test["return"], periods=365)*100, 2)}%',
          f'MDD:\t\t\t{round(qs.stats.max_drawdown(test["return"])*100, 2)}%',
          f'Calmar:\t\t\t{round(-qs.stats.cagr(test["return"], periods=365)/qs.stats.max_drawdown(test["return"]), 2)}',
          sep='\n')
    
def generate_metrics_indexed(df, title, periods):
    test = df.copy()
    DAYS_PER_YEAR = 365
    print(f'\033[92m\n\033[01m{"PERFORMANCE METRICS" if not title else title}\033[0m',
          f'Cumulative Return:\t{round(((test["return"].add(1).cumprod().sub(1)).iloc[-1])*100,2)}%',
          f'Sharpe:\t\t\t{round(qs.stats.sharpe(test["return"], periods=periods), 2)}',
          f'CAGR:\t\t\t{round(qs.stats.cagr(test["return"], periods=365)*100, 2)}%',
          f'MDD:\t\t\t{round(qs.stats.max_drawdown(test["return"])*100, 2)}%',
          f'Calmar:\t\t\t{round(-qs.stats.cagr(test["return"], periods=365)/qs.stats.max_drawdown(test["return"]), 2)}',
          sep='\n')
    
def plots_pnl(df,title):
    # Plot the PnL curve
    fig, ax = plt.subplots(figsize=(24, 16))
    result = df.copy()
    result['return'] = result['return'].fillna(0)
    plt.plot(result['return'], color='blue')
    ax.set_facecolor('white')
    plt.xlabel('Time')
    plt.ylabel('PnL')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'{output_path}/{title}_pnl.png', format='png')
    plt.show()

def plots_cum_return(df,title, scale):
    fig, ax = plt.subplots(figsize=(24, 16))
    result = df.copy()
    if scale == 'normal':
        result['Total_return'] = result['return'].add(1).cumprod().sub(1)
        result['Total_Long_return'] = result['long_return'].add(1).cumprod().sub(1).ffill()
        result['Total_Short_return'] = result['short_return'].add(1).cumprod().sub(1).ffill()
        plt.plot(result['Total_return'], label = 'Total Return', color='blue')
        plt.plot(result['Total_Long_return'], label = 'Long Return', color='red')
        plt.plot(result['Total_Short_return'], label = 'Short Return', color='green')
    if scale == 'log':
        result['Total_return'] = np.log(result['return'].add(1).cumprod())
        result['Total_Long_return'] = np.log(result['long_return'].add(1).cumprod()).ffill()
        result['Total_Short_return'] = np.log(result['short_return'].add(1).cumprod()).ffill()
        plt.plot(result['Total_return'], label = 'Total Return (log_scale)', color='blue')
        plt.plot(result['Total_Long_return'], label = 'Long Return (log_scale)', color='red')
        plt.plot(result['Total_Short_return'], label = 'Short Return (log_scale)', color='green')        
    # Adjust font sizes
    ax.set_facecolor('white')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/{title}_long_short.png', format='png')
    plt.show()

def plots_Equity(df,df2,title, scale):
    fig, ax = plt.subplots(figsize=(24, 16))
    result = df.copy()
    benchmark = df2.copy()
    if scale == 'normal':
        result['Total_return'] = result['return'].add(1).cumprod().sub(1)
        benchmark['return'] = benchmark['close'].pct_change()
        benchmark['cum_return'] = (benchmark['return'].add(1).cumprod() - 1)
        plt.plot(result['Total_return'], label = 'Equity',color='blue')
        plt.plot(benchmark['cum_return'], label = 'Benchmark', color='red')
    if scale == 'log':
        result['Total_return'] = np.log(result['return'].add(1).cumprod())
        benchmark['return'] = benchmark['close'].pct_change()
        benchmark['cum_return'] = np.log(benchmark['return'].add(1).cumprod())
        plt.plot(result['Total_return'], label = 'Equity (log_scale)',color='blue')
        plt.plot(benchmark['cum_return'], label = 'Benchmark (log_scale)', color='red')
    ax.set_facecolor('white')
    # Add legends
    # Adjust plot aesthetics
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_path}/{title}_equity_vs_benchmark.png', format='png')
    plt.show()

def generate_heatmap(train_result_dict, test_result_dict, full_result_dict):
    sharpe = {'Train Set': train_result_dict, 'Test Set': test_result_dict, 'Full Set': full_result_dict}
    sns.set(font_scale=0.8)
    sns.set(rc={'figure.figsize': (24, 16)})
    for dataset_name, dataset in sharpe.items():
        plt.figure()
        ser = pd.Series(list(dataset.values()), index=pd.MultiIndex.from_tuples(dataset.keys()))
        sharpe = ser.unstack().fillna(0)
        sns.heatmap(sharpe, annot=True, cmap="crest", fmt=".2f").set_title(f'{dataset_name}_heatmap')
        plt.xlabel('z_score')
        plt.ylabel('Period')
        plt.show()
        plt.close()


def generate_3D_heatmap(data_dict, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    parameters = list(data_dict.keys())
    sharpe_ratios = [data_dict[key] for key in parameters]

    x = [param[0] for param in parameters]
    y = [param[1] for param in parameters]
    z = [param[2] for param in parameters]

    ax.scatter(x, y, z, c=sharpe_ratios, cmap='viridis')

    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Parameter 3')
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(ax.scatter(x, y, z, c=sharpe_ratios, cmap='viridis'))
    cbar.set_label('Sharpe Ratio')

    plt.show()
