import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pandas_ta as ta
import matplotlib.pyplot as plt
import warnings
import quantstats as qs
from utils.simulation import simulation
warnings.filterwarnings("ignore")


def Bbands_at_band(df,prd,z_score,Type,ind,strat):
    df_test = df.copy()
    prd = int(prd)
    df_test.loc[:,'z_score'] = (df_test[f'{ind}'] - df_test[f'{ind}'].rolling(window=prd).mean()) / (df_test[f'{ind}'].rolling(window=prd).std())

    if strat == 'Momentum':
        df_test.loc[df_test['z_score'] > z_score, 'signal'] = 1         # condition to Long position
        df_test.loc[df_test['z_score'] < -z_score, 'signal'] = -1         # condition to Long position
        df_test.loc[(df_test['z_score'] < z_score) & (df_test['z_score'].shift(1) > z_score) & (df_test['z_score'] < z_score) & (df_test['z_score'] > -z_score), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['z_score'] > -z_score) & (df_test['z_score'].shift(1) < -z_score) & (df_test['z_score'] > -z_score) & (df_test['z_score'] < z_score), 'signal'] = 0         # condition to close position

    elif strat == 'Reversion':
        df_test.loc[df_test['z_score'] > z_score, 'signal'] = -1         # condition to Long position
        df_test.loc[df_test['z_score'] < -z_score, 'signal'] = 1         # condition to Long position
        df_test.loc[(df_test['z_score'] < z_score) & (df_test['z_score'].shift(1) > z_score) & (df_test['z_score'] < z_score) & (df_test['z_score'] > -z_score), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['z_score'] > -z_score) & (df_test['z_score'].shift(1) < -z_score) & (df_test['z_score'] > -z_score) & (df_test['z_score'] < z_score), 'signal'] = 0         # condition to close position
    else:
        print('invalid strat')
    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    result = simulation(df_test,Type)
    return result

def Bbands_flat_zero(df,prd,z_score,Type,ind,strat):
    df_test = df.copy()
    prd = int(prd)
    df_test[f'{ind}'] = df_test[f'{ind}'].astype(float)
    df_test.loc[:,'z_score'] = (df_test[f'{ind}'] - df_test[f'{ind}'].rolling(window=prd).mean()) / (df_test[f'{ind}'].rolling(window=prd).std())

    if strat == 'Momentum':
        df_test.loc[df_test['z_score'] > z_score, 'signal'] = 1         # condition to Long position
        df_test.loc[df_test['z_score'] < -z_score, 'signal'] = -1         # condition to Long position
        df_test.loc[(df_test['z_score'] < 0) & (df_test['z_score'].shift(1) > 0) & (df_test['z_score'] < z_score) & (df_test['z_score'] > -z_score), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['z_score'] > 0) & (df_test['z_score'].shift(1) < 0) & (df_test['z_score'] > -z_score) & (df_test['z_score'] < z_score), 'signal'] = 0         # condition to close position

    elif strat == 'Reversion':
        df_test.loc[df_test['z_score'] > z_score, 'signal'] = -1         # condition to Long position
        df_test.loc[df_test['z_score'] < -z_score, 'signal'] = 1         # condition to Long position
        df_test.loc[(df_test['z_score'] < 0) & (df_test['z_score'].shift(1) > 0) & (df_test['z_score'] < z_score) & (df_test['z_score'] > -z_score), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['z_score'] > 0) & (df_test['z_score'].shift(1) < 0) & (df_test['z_score'] > -z_score) & (df_test['z_score'] < z_score), 'signal'] = 0         # condition to close position
    else:
        print('invalid strat')
    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    result = simulation(df_test,Type)
    return result

def SBT_KAMA(df,prd,mult,kama_prd,Type,ind,method): #ZigZag Version

    performance_df = pd.DataFrame(index=df.index)
    performance_df[df.columns] = df
    performance_df['open'] = performance_df['open'].astype(float)
    performance_df['close'] = performance_df['close'].astype(float)
    performance_df['high'] = performance_df['high'].astype(float)
    performance_df['low'] = performance_df['low'].astype(float)

    performance_df['ROC'] = performance_df['close'].pct_change()

    change = abs(performance_df['close']-performance_df['close'].shift(kama_prd))
    volatility = (abs(performance_df['close']-performance_df['close'].shift())).rolling(kama_prd).sum()
    er = change/volatility

    #Smoothing Constant
    sc_fatest = 2/(2 + 1)
    sc_slowest = 2/(30 + 1)
    sc= (er * (sc_fatest - sc_slowest) + sc_slowest)**2

    kama = np.zeros_like(performance_df['close'])
    kama[kama_prd-1] = performance_df['close'].iloc[kama_prd-1]       
    for i in range(kama_prd, len(performance_df)):
        kama[i] = kama[i-1] + sc.iloc[i] * (performance_df['close'].iloc[i] - kama[i-1])
        kama[kama == 0]=np.nan
    performance_df['KAMA']=kama

    performance_df['sbt'] = None
    performance_df.loc[performance_df.index[prd-2],'sbt'] = 0
    performance_df.loc[performance_df.index[prd-1],'sbt'] = 0

    # Calculate the Bollinger Bands
    performance_df.loc[:,'bbup'] = performance_df['high'].rolling(window=prd).mean() + mult * performance_df['high'].rolling(window=prd).std()
    performance_df.loc[:,'bbdn'] = performance_df['low'].rolling(window=prd).mean() - mult * performance_df['low'].rolling(window=prd).std()

    performance_df['signal'] = 0
    performance_df['sbt_signal'] = 0
    green_triangle = False
    red_triangle = False
    for date in performance_df.index[prd-1:]:
        time_diff = date - performance_df.index[performance_df.index.get_loc(date) - 1]
        
        # Update SBT
        if pd.notnull(performance_df.loc[date - time_diff, 'sbt']) & (performance_df.loc[date, 'close'] > performance_df.loc[date - time_diff, 'sbt']):
            performance_df.loc[date, 'signal'] = performance_df.loc[date - time_diff, 'signal']
            performance_df.loc[date, 'sbt'] = max(performance_df.loc[date, 'bbdn'], performance_df.loc[date - time_diff, 'sbt'])
        elif pd.notnull(performance_df.loc[date - time_diff, 'sbt']) & (performance_df.loc[date, 'close'] < performance_df.loc[date - time_diff, 'sbt']):
            performance_df.loc[date, 'signal'] = performance_df.loc[date - time_diff, 'signal']
            performance_df.loc[date, 'sbt'] = min(performance_df.loc[date, 'bbup'], performance_df.loc[date - time_diff, 'sbt'])
        else:
            performance_df.loc[date, 'sbt'] = 0

        # Check Close Position
        if ((performance_df.loc[date, 'signal'] == 1.0) & (performance_df.loc[date - time_diff, 'close'] > performance_df.loc[date, 'KAMA']) & (performance_df.loc[date, 'close'] < performance_df.loc[date, 'KAMA'])):
            performance_df.loc[date, 'signal'] = 0.0
            green_triangle = False

        if ((performance_df.loc[date, 'signal'] == -1.0) & (performance_df.loc[date - time_diff, 'close'] < performance_df.loc[date, 'KAMA']) & (performance_df.loc[date, 'close'] > performance_df.loc[date, 'KAMA'])):
            performance_df.loc[date, 'signal'] = 0.0
            red_triangle = False

        # Check Triangle
        if (performance_df.loc[date, 'close'] > performance_df.loc[date - time_diff, 'sbt']) & (performance_df.loc[date - time_diff, 'close'] < performance_df.loc[date - time_diff, 'sbt']) and pd.notnull(performance_df.loc[date - time_diff, 'sbt']):
            if pd.notnull(performance_df.loc[date, 'bbdn']):
                green_triangle = True
                performance_df.loc[date, 'sbt_signal'] = 1.0
                performance_df.loc[date, 'sbt'] = performance_df.loc[date, 'bbdn']

        if (performance_df.loc[date, 'close'] < performance_df.loc[date - time_diff, 'sbt']) & (performance_df.loc[date - time_diff, 'close'] > performance_df.loc[date - time_diff, 'sbt']) and pd.notnull(performance_df.loc[date - time_diff, 'sbt']):
            if pd.notnull(performance_df.loc[date, 'bbup']):
                red_triangle = True
                performance_df.loc[date, 'sbt_signal'] = -1.0
                performance_df.loc[date, 'sbt'] = performance_df.loc[date, 'bbup']

        # Check signal
        if (green_triangle == True and (performance_df.loc[date, 'close'] > performance_df.loc[date, 'KAMA'])):
            performance_df.loc[date, 'signal'] = 1.0
            green_triangle = False

        if (red_triangle == True and (performance_df.loc[date, 'close'] < performance_df.loc[date, 'KAMA'])):
            performance_df.loc[date, 'signal'] = -1.0
            red_triangle = False

    result = simulation(performance_df,Type)
    return result

def SBT_Test(df,prd,mult,Type): #SBT(1,-1)
    if (Type == 'Spot'):
        commission = 3 / 10000 # from bps to decimal
    elif (Type == 'Futures'):
        commission = 6 / 10000 # 6bps to decimal
    else:
        raise ValueError('market must be either "SPOT" or "FUTURES"')
    performance_df = pd.DataFrame(index=df.index)
    performance_df[df.columns] = df
    performance_df['ROC'] = performance_df['close'].pct_change()
    performance_df['sbt'] = None
    performance_df.loc[prd-2,'sbt'] = 0


    # Calculate the Bollinger Bands
    performance_df.loc[:,'bbup'] = performance_df['high'].rolling(window=prd).mean() + mult * performance_df['high'].rolling(window=prd).std()
    performance_df.loc[:,'bbdn'] = performance_df['low'].rolling(window=prd).mean() - mult * performance_df['low'].rolling(window=prd).std()

    performance_df['signal'] = 0
    # performance_df['TP'] = 0
    # performance_df['signal point'] = 0
    for i in range(prd-1, len(performance_df)):
        # Update SBT
        if pd.notnull(performance_df.loc[i-1, 'sbt']) & (performance_df.loc[i, 'close'] > performance_df.loc[i-1, 'sbt']):
            performance_df.loc[i,'signal'] = performance_df.loc[i-1,'signal']
            performance_df.loc[i, 'sbt'] = max(performance_df.loc[i, 'bbdn'], performance_df.loc[i-1, 'sbt'])
        elif pd.notnull(performance_df.loc[i-1, 'sbt']) & (performance_df.loc[i, 'close'] < performance_df.loc[i-1, 'sbt']):
            performance_df.loc[i,'signal'] = performance_df.loc[i-1,'signal']
            performance_df.loc[i, 'sbt'] = min(performance_df.loc[i, 'bbup'], performance_df.loc[i-1, 'sbt'])
        else:
            performance_df.loc[i,'sbt'] = 0

        # Check signal
        if (performance_df.loc[i,'close'] > performance_df.loc[i-1,'sbt']) & (performance_df.loc[i-1,'close'] < performance_df.loc[i-1,'sbt']) and pd.notnull(performance_df.loc[i-1, 'sbt']):
        # if crossover(performance_df.loc[i,'close'],performance_df.loc[i-1,'close'],performance_df.loc[i-1,'sbt'])and pd.notnull(performance_df.loc[i-1, 'sbt']):
            if pd.notnull(performance_df.loc[i, 'bbdn']):
                performance_df.loc[i,'signal'] = 1.0
                performance_df.loc[i,'sbt'] = performance_df.loc[i,'bbdn']
            
        if (performance_df.loc[i,'close'] < performance_df.loc[i-1,'sbt']) & (performance_df.loc[i-1,'close'] > performance_df.loc[i-1,'sbt']) and pd.notnull(performance_df.loc[i-1, 'sbt']):
        # if crossunder(performance_df.loc[i,'close'],performance_df.loc[i-1,'close'],performance_df.loc[i-1,'sbt'])and pd.notnull(performance_df.loc[i-1, 'sbt']):
            if pd.notnull(performance_df.loc[i, 'bbup']):
                performance_df.loc[i,'signal'] = -1.0
                performance_df.loc[i,'sbt'] = performance_df.loc[i,'bbup']

    result = simulation(performance_df,Type)
    return result

def TBSV2(df,prd,z_score,Type,ind,method):

    performance_df = df.copy()
    prd = int(prd)
    performance_df['close'] = performance_df['close'].astype(float)

    performance_df.loc[:,'z_close'] = (performance_df['close'] - performance_df['close'].rolling(window=prd).mean()) / (performance_df['close'].rolling(window=prd).std())
    performance_df.loc[:,'z_ratio'] = (performance_df['log_tbsv_ratio'] - performance_df['log_tbsv_ratio'].rolling(window=prd).mean()) / (performance_df['log_tbsv_ratio'].rolling(window=prd).std())
    performance_df['z_total'] = (performance_df['z_close'] + performance_df['z_ratio']) / 2                                                                                                                              
    
    if method == 'Momentum':
        performance_df.loc[performance_df['z_total'] > z_score, 'signal'] = 1         # condition to Long position
        performance_df.loc[performance_df['z_total'] < -z_score, 'signal'] = -1         # condition to Short position

        performance_df.loc[(performance_df['z_total'] < 0) & (performance_df['z_total'].shift(1) > 0) & (performance_df['z_total'] < z_score) & (performance_df['z_total'] > -z_score), 'signal'] = 0         # condition to close position
        performance_df.loc[(performance_df['z_total'] > 0) & (performance_df['z_total'].shift(1) < 0) & (performance_df['z_total'] > -z_score) & (performance_df['z_total'] < z_score), 'signal'] = 0         # condition to close position
    
    elif method == 'Reversion':
        performance_df.loc[performance_df['z_total'] > z_score, 'signal'] = -1         # condition to Long position
        performance_df.loc[performance_df['z_total'] < -z_score, 'signal'] = 1         # condition to Short position

        performance_df.loc[(performance_df['z_total'] < 0) & (performance_df['z_total'].shift(1) > 0) & (performance_df['z_total'] < z_score) & (performance_df['z_total'] > -z_score), 'signal'] = 0         # condition to close position
        performance_df.loc[(performance_df['z_total'] > 0) & (performance_df['z_total'].shift(1) < 0) & (performance_df['z_total'] > -z_score) & (performance_df['z_total'] < z_score), 'signal'] = 0         # condition to close position
    
    else:
        print('invalid method')

    performance_df['signal'] = performance_df['signal'].mul(1).ffill().fillna(0)
    
    result = simulation(performance_df,Type)
    return result

def calculate_PPO(df,short_period,long_period,Type,ind,method):
    # Calculate short-term EMA
    df_test = df.copy()
    short_period = int(short_period)
    long_period = int(long_period)
    short_ema = df_test[f'{ind}'].ewm(span=short_period, adjust=False).mean()
    # Calculate long-term EMA
    long_ema = df_test[f'{ind}'].ewm(span=long_period, adjust=False).mean()
    # Calculate PPO
    df_test['PPO'] = ((short_ema - long_ema) / long_ema) * 100
    df_test.loc[df_test['PPO'] > 0, 'signal'] = 1         # condition to Long position
    df_test.loc[df_test['PPO'] < 0, 'signal'] = -1 
    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    result = simulation(df_test,Type)
    return result


def marketSpread(df, x, y,Type,ind,method):
    # generate indicator
    df_test = df.copy()
    # df_test['futures-spot_close'] = (df_test['futures_close']/df_test['spot_close'])-1
    df_test.ta.ema(close=df_test[f'{ind}'], length = y, append=True, col_names=("EMA"))
    df_test.ta.sma(close=df_test['EMA'], length = x, append=True, col_names = ("MA"))

    df_test.loc[df_test['EMA'] <= df_test['MA'], 'signal'] = 1            # condition to long
    df_test.loc[df_test['EMA'] > df_test['MA'], 'signal'] = -1            # condition to short
    
    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    
    result = simulation(df_test,Type)
    return result


def tbsv_ema(df, n, z, Type, ind, method):
    df_test = df.copy()
    n = int(n)
    df_test.ta.ema(close=df_test[f'{ind}'], length=n, append=True, col_names=('ema'))
    df_test.ta.zscore(close=df_test['ema'], length=n, append=True, col_names=('ZSCR_LOG_TBSV'))
    df_test.ta.zscore(close=df_test['close'], length=n, append=True, col_names=('ZSCR_CLOSE'))

    df_test['avg'] = (df_test['ZSCR_LOG_TBSV'] + df_test['ZSCR_CLOSE']) / 2

    df_test['long_signal']=df_test.ta.xsignals(signal=df_test['avg'], xa=z, xb=0, long=True, above=True).TS_Trends
    df_test['short_signal']=df_test.ta.xsignals(signal=df_test['avg'], xa=-z, xb=0, long=True, above=False).TS_Trends
    df_test['signal'] = df_test['long_signal'] - df_test['short_signal']
    df_test.loc[df_test['avg'].isna(), 'signal'] = 0

    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)

    result = simulation(df_test,Type)
    # print(df.to_string())

    return result

def tbsv_wma(df, n, z, Type, ind, method):
    df_test = df.copy()
    n = int(n)
    df_test.ta.wma(close=df_test[f'{ind}'], length=n, append=True, col_names=('wma'))
    df_test.ta.zscore(close=df_test['wma'], length=n, append=True, col_names=('ZSCR_LOG_TBSV'))
    df_test.ta.zscore(close=df_test['close'], length=n, append=True, col_names=('ZSCR_CLOSE'))

    df_test['avg'] = (df_test['ZSCR_LOG_TBSV'] + df_test['ZSCR_CLOSE']) / 2

    df_test['long_signal']=df_test.ta.xsignals(signal=df_test['avg'], xa=z, xb=0, long=True, above=True).TS_Trends
    df_test['short_signal']=df_test.ta.xsignals(signal=df_test['avg'], xa=-z, xb=0, long=True, above=False).TS_Trends
    df_test['signal'] = df_test['long_signal'] - df_test['short_signal']
    df_test.loc[df_test['avg'].isna(), 'signal'] = 0

    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)

    result = simulation(df_test,Type)
    # print(df.to_string())
    
    return result

def tbsv_sma(df, n, z, Type, ind, method):
    df_test = df.copy()
    n = int(n)
    df_test.ta.sma(close=df_test[f'{ind}'], length=n, append=True, col_names=('sma'))
    df_test.ta.zscore(close=df_test['sma'], length=n, append=True, col_names=('ZSCR_LOG_TBSV'))
    df_test.ta.zscore(close=df_test['close'], length=n, append=True, col_names=('ZSCR_CLOSE'))

    df_test['avg'] = (df_test['ZSCR_LOG_TBSV'] + df_test['ZSCR_CLOSE']) / 2

    df_test['long_signal']=df_test.ta.xsignals(signal=df_test['avg'], xa=z, xb=0, long=True, above=True).TS_Trends
    df_test['short_signal']=df_test.ta.xsignals(signal=df_test['avg'], xa=-z, xb=0, long=True, above=False).TS_Trends
    df_test['signal'] = df_test['long_signal'] - df_test['short_signal']
    df_test.loc[df_test['avg'].isna(), 'signal'] = 0

    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)

    result = simulation(df_test,Type)
    # print(df.to_string())

    return result


def coinbase_long_only(df,prd,z_score,Type,ind,method):
    df_test = df.copy()
    prd = int(prd)
    df_test.loc[:,'z_score'] = (df_test[f'{ind}'] - df_test[f'{ind}'].rolling(window=prd).mean()) / (df_test[f'{ind}'].rolling(window=prd).std())
    df_test.loc[df_test['z_score'] > z_score, 'signal'] = 1         # condition to Long position

    df_test.loc[(df_test['z_score'] < z_score) & (df_test['z_score'].shift(1) > z_score), 'signal'] = 0         # condition to close position
    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    result = simulation(df_test,Type)
    return result

def coinbase_short_only(df,prd,z_score,Type,ind,method):
    df_test = df.copy()
    prd = int(prd)
    df_test.loc[:,'z_score'] = (df_test[f'{ind}'] - df_test[f'{ind}'].rolling(window=prd).mean()) / (df_test[f'{ind}'].rolling(window=prd).std())
    df_test.loc[df_test['z_score'] < -z_score, 'signal'] = -1         # condition to Long position

    df_test.loc[(df_test['z_score'] > 0) & (df_test['z_score'].shift(1) < 0), 'signal'] = 0         # condition to close position
    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    result = simulation(df_test,Type)
    return result

def ttm_squeeze(merge_df,prd,z,Type,ind,method):
    prd = int(prd)
    bband = ta.bbands(merge_df['close'],prd)
    kc = ta.kc(merge_df['high'],merge_df['low'],merge_df['close'],z)
    test = ta.squeeze(merge_df['high'],merge_df['low'],merge_df['close'],bb_length=prd,kc_length=z)
    test2 = pd.merge(bband,kc,left_index=True,right_index=True)
    test3 = pd.merge(test2,test,left_index=True,right_index=True)
    df_test = pd.merge(merge_df,test3,left_index=True,right_index=True)
    df_test.loc[(df_test[f'SQZ_{prd}_2.0_{z}_1.5'] > 0) & (df_test['SQZ_ON'] == 1) ,'signal'] = 1
    df_test.loc[(df_test[f'SQZ_{prd}_2.0_{z}_1.5'] < 0) & (df_test['SQZ_ON'] == 1) ,'signal'] = -1

    df_test.loc[((df_test['close'] > df_test[f'BBU_{prd}_2.0'].shift()) & (df_test['SQZ_OFF'] == 1) & df_test['signal'] == 1),'signal'] = 0
    df_test.loc[((df_test['close'] < df_test[f'BBL_{prd}_2.0'].shift()) & (df_test['SQZ_OFF'] == 1) & df_test['signal'] == -1),'signal'] = 0

    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    result = simulation(df_test,Type)
    return result

def ema_bbands_flat_zero(df, n, z, Type,ind,strat):
    df_test = df.copy()
    n = int(n)
    df_test['ema'] = df_test[f'{ind}'].ewm(span=n).mean()
    df_test['z_log_ratio'] = (df_test[f'{ind}'] - df_test['ema']) / df_test['ema'].rolling(window=n).std()
    df_test['z_close'] = (df_test['close'] - df_test['close'].rolling(window=n).mean()) / df_test['close'].rolling(window=n).std()
    df_test['z_score'] = (df_test['z_close'] + df_test['z_log_ratio']) / 2

    if strat == 'Momentum':
        df_test.loc[df_test['z_score'] > z, 'signal'] = 1         # condition to Long position
        df_test.loc[df_test['z_score'] < -z, 'signal'] = -1         # condition to Long position
        df_test.loc[(df_test['z_score'] < 0) & (df_test['z_score'].shift(1) > 0) & (df_test['z_score'] < z) & (df_test['z_score'] > -z), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['z_score'] > 0) & (df_test['z_score'].shift(1) < 0) & (df_test['z_score'] > -z) & (df_test['z_score'] < z), 'signal'] = 0         # condition to close position

    elif strat == 'Reversion':
        df_test.loc[df_test['z_score'] > z, 'signal'] = -1         # condition to Long position
        df_test.loc[df_test['z_score'] < -z, 'signal'] = 1         # condition to Long position
        df_test.loc[(df_test['z_score'] < 0) & (df_test['z_score'].shift(1) > 0) & (df_test['z_score'] < z) & (df_test['z_score'] > -z), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['z_score'] > 0) & (df_test['z_score'].shift(1) < 0) & (df_test['z_score'] > -z) & (df_test['z_score'] < z), 'signal'] = 0         # condition to close position
    else:
        print('invalid strat')
    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    result = simulation(df_test,Type)
    return result

def RSI(df, prd, delta, Type, ind, strat):
    df_test = df.copy()
    prd = int(prd)
    df_test['RSI'] = ta.rsi(df_test[f'{ind}'],prd)
    if strat == 'Momentum':
        df_test.loc[df_test['RSI'] > (50+delta), 'signal'] = 1         # condition to Long position
        df_test.loc[df_test['RSI'] < (50-delta), 'signal'] = -1         # condition to Long position
        df_test.loc[(df_test['RSI'] > 50) & (df_test['RSI'].shift(1) < 50) & (df_test['RSI'] < (50+delta)) & (df_test['RSI'] > (50-delta)), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['RSI'] < 50) & (df_test['RSI'].shift(1) > 50) & (df_test['RSI'] > (50-delta)) & (df_test['RSI'] < (50+delta)), 'signal'] = 0         # condition to close position
    
    elif strat == 'Reversion':
        df_test.loc[df_test['RSI'] > (50+delta), 'signal'] = -1         # condition to Long position
        df_test.loc[df_test['RSI'] < (50-delta), 'signal'] = 1         # condition to Long position
        df_test.loc[(df_test['RSI'] > 50) & (df_test['RSI'].shift(1) < 50) & (df_test['RSI'] < (50+delta)) & (df_test['RSI'] > (50-delta)), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['RSI'] < 50) & (df_test['RSI'].shift(1) > 50) & (df_test['RSI'] > (50-delta)) & (df_test['RSI'] < (50+delta)), 'signal'] = 0         # condition to close position
    
    else:
        print('invalid strat')
    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    result = simulation(df_test,Type)
    return result

def ma_cross(df, short_window, long_window, Type, ind, strat):
    df_test = df.copy()
    short_window = int(short_window)
    long_window = int(long_window)
    df_test['short_ema'] = df_test[f'{ind}'].ewm(span=short_window, adjust=False).mean()
    df_test['long_ema'] = df_test[f'{ind}'].ewm(span=long_window, adjust=False).mean()
    df_test['esma_spread'] = df_test['short_ema'] - df_test['long_ema']

    if strat == 'Momentum':
        df_test['signal'] = 0
        df_test.loc[df_test['esma_spread'] > 0, 'signal'] = 1  # Long position
        df_test.loc[df_test['esma_spread'] < 0, 'signal'] = -1  # Short position
        df_test.loc[
            (df_test['esma_spread'] < 0) & (df_test['esma_spread'].shift(1) > 0) |
            (df_test['esma_spread'] > 0) & (df_test['esma_spread'].shift(1) < 0), 'signal'] = 0  # Close position
    elif strat == 'Reversion':
        df_test['signal'] = 0
        df_test.loc[df_test['esma_spread'] > 0, 'signal'] = -1  # Short position
        df_test.loc[df_test['esma_spread'] < 0, 'signal'] = 1  # Long position
        df_test.loc[
            (df_test['esma_spread'] < 0) & (df_test['esma_spread'].shift(1) > 0) |
            (df_test['esma_spread'] > 0) & (df_test['esma_spread'].shift(1) < 0), 'signal'] = 0  # Close position
    else:
        print('Invalid strategy')

    df_test['signal'] = df_test['signal'].ffill().fillna(0)
    result = simulation(df_test, Type)
    return result

def esma_spread(df, window, threshold, Type, ind, strat):
    df_test = df.copy()
    window = int(window)
    df_test['ema'] = df_test[f'{ind}'].ewm(span=window, adjust=False).mean()
    df_test['sma'] = df_test[f'{ind}'].rolling(window=window).mean()
    df_test['esma_spread'] = (df_test['ema'] - df_test['sma'])
    ## normalise the esma spread with the same rolling window
    df_test['normalised_spread'] = (df_test['esma_spread'] - df_test['esma_spread'].rolling(window=window).mean()) / df_test['esma_spread'].rolling(window=window).std()

    if strat == 'Momentum':
        df_test['signal'] = 0
        df_test.loc[df_test['normalised_spread'] > threshold, 'signal'] = 1  # Long position
        df_test.loc[df_test['normalised_spread'] < threshold, 'signal'] = -1  # Short position

    elif strat == 'Reversion':
        df_test['signal'] = 0
        df_test.loc[df_test['normalised_spread'] > threshold, 'signal'] = -1  # Short position
        df_test.loc[df_test['normalised_spread'] < threshold, 'signal'] = 1  # Long position

    else:
        print('Invalid strategy')

    df_test['signal'] = df_test['signal'].ffill().fillna(0)
    result = simulation(df_test, Type)
    return result

def min_max_norm(df, prd, z_score, Type, ind, strat):
    df_test = df.copy()
    prd = int(prd)
    df_test['min'] = df_test[f'{ind}'].rolling(window=prd).min()
    df_test['max'] = df_test[f'{ind}'].rolling(window=prd).max()
    df_test['norm'] = (df_test[f'{ind}'] - df_test['min']) / (df_test['max'] - df_test['min'])
    df_test['z_score'] = (df_test['norm'] - df_test['norm'].rolling(window=prd).mean()) / df_test['norm'].rolling(window=prd).std()

    if strat == 'Momentum':
        df_test.loc[df_test['z_score'] > z_score, 'signal'] = 1  # Long position
        df_test.loc[df_test['z_score'] < -z_score, 'signal'] = -1
        df_test.loc[(df_test['z_score'] < 0) & (df_test['z_score'].shift(1) > 0) & (df_test['z_score'] < z_score) & (df_test['z_score'] > -z_score), 'signal'] = 0
        df_test.loc[(df_test['z_score'] > 0) & (df_test['z_score'].shift(1) < 0) & (df_test['z_score'] > -z_score) & (df_test['z_score'] < z_score), 'signal'] = 0

    elif strat == 'Reversion':
        df_test.loc[df_test['z_score'] > z_score, 'signal'] = -1  # Short position
        df_test.loc[df_test['z_score'] < -z_score, 'signal'] = 1
        df_test.loc[(df_test['z_score'] < 0) & (df_test['z_score'].shift(1) > 0) & (df_test['z_score'] < z_score) & (df_test['z_score'] > -z_score), 'signal'] = 0
        df_test.loc[(df_test['z_score'] > 0) & (df_test['z_score'].shift(1) < 0) & (df_test['z_score'] > -z_score) & (df_test['z_score'] < z_score), 'signal'] = 0

    else:
        print('Invalid strategy')
    
    df_test['signal'] = df_test['signal'].ffill().fillna(0)
    result = simulation(df_test, Type)
    return result


def robust_scaler(df, prd, z_score, Type, ind, strat):
    df_test = df.copy()
    prd = int(prd)  
    df_test['median'] = df_test[f'{ind}'].rolling(window=prd).median()
    df_test['iqr'] = df_test[f'{ind}'].rolling(window=prd).quantile(0.75) - df_test[f'{ind}'].rolling(window=prd).quantile(0.25)
    df_test['robust'] = (df_test[f'{ind}'] - df_test['median']) / df_test['iqr']
    df_test['z_score'] = (df_test['robust'] - df_test['robust'].rolling(window=prd).mean()) / df_test['robust'].rolling(window=prd).std()

    if strat == 'Momentum':
        df_test.loc[df_test['z_score'] > z_score, 'signal'] = 1  # Long position
        df_test.loc[df_test['z_score'] < -z_score, 'signal'] = -1
        df_test.loc[(df_test['z_score'] < 0) & (df_test['z_score'].shift(1) > 0) & (df_test['z_score'] < z_score) & (df_test['z_score'] > -z_score), 'signal'] = 0
        df_test.loc[(df_test['z_score'] > 0) & (df_test['z_score'].shift(1) < 0) & (df_test['z_score'] > -z_score) & (df_test['z_score'] < z_score), 'signal'] = 0

    elif strat == 'Reversion':
        df_test.loc[df_test['z_score'] > z_score, 'signal'] = -1  # Short position
        df_test.loc[df_test['z_score'] < -z_score, 'signal'] = 1
        df_test.loc[(df_test['z_score'] < 0) & (df_test['z_score'].shift(1) > 0) & (df_test['z_score'] < z_score) & (df_test['z_score'] > -z_score), 'signal'] = 0
        df_test.loc[(df_test['z_score'] > 0) & (df_test['z_score'].shift(1) < 0) & (df_test['z_score'] > -z_score) & (df_test['z_score'] < z_score), 'signal'] = 0

    else:
        print('Invalid strategy')
    
    df_test['signal'] = df_test['signal'].ffill().fillna(0)
    result = simulation(df_test, Type)
    return result

def divergence(df, prd, z_score, Type, ind, strat):
    df_test = df.copy()
    prd = int(prd)
    df_test['price_low'] = df_test['close'].rolling(window=prd).min()
    df_test['value_low'] = df_test[f'{ind}'].rolling(window=prd).min()


    bullish_divergence = (df_test['close'] < df_test['price_low'].shift(1)) & (df_test[f'{ind}'] > 
                        df_test['value_low'].shift(1))
    df_test.loc[bullish_divergence, 'signal'] = -1

    df_test['price_high'] = df_test['close'].rolling(window=prd).max()
    df_test['value_high'] = df_test[f'{ind}'].rolling(window=prd).max()
    bearish_divergence = (df_test['close'] > df_test['price_high'].shift(1)) & (df_test[f'{ind}'] <
                        df_test['value_high'].shift(1))
    
    df_test.loc[bearish_divergence , 'signal'] =    1
    df_test['signal'] = df_test['signal'].ffill().fillna(0)
    result = simulation(df_test, Type)
    return result

def RSI(df,prd,threshold,Type,ind,strat):
    df_test = df.copy()
    prd = int(prd)
    df_test.loc[:,'RSI'] = ta.rsi(df_test[f'{ind}'],prd)
    df_test.loc[:,'signal'] = 0

    if strat == 'Momentum':
        df_test.loc[df_test['RSI'] > 50+threshold, 'signal'] = 1         # condition to Long position
        df_test.loc[df_test['RSI'] < 50-threshold, 'signal'] = -1         # condition to Long position
        df_test.loc[(df_test['RSI'] < 50) & (df_test['RSI'].shift(1) > 50), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['RSI'] > 50) & (df_test['RSI'].shift(1) < 50), 'signal'] = 0         # condition to close position

    elif strat == 'Reversion':
        df_test.loc[df_test['RSI'] > 50+threshold, 'signal'] = -1         # condition to Long position
        df_test.loc[df_test['RSI'] < 50-threshold, 'signal'] = 1         # condition to Long position
        df_test.loc[(df_test['RSI'] < 50) & (df_test['RSI'].shift(1) > 50), 'signal'] = 0         # condition to close position
        df_test.loc[(df_test['RSI'] > 50) & (df_test['RSI'].shift(1) < 50), 'signal'] = 0         # condition to close position
    else:
        print('invalid strat')
    df_test['signal'] = df_test['signal'].mul(1).ffill().fillna(0)
    result = simulation(df_test,Type) 
    return result