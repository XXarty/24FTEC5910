import datetime
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import quantstats as qs
from utils.simulation import simulation

warnings.filterwarnings("ignore")

class HeatmapAnalyzer:
    def __init__(self, prd, z_score,prd2,step1,step2,step3,df, start, end, ratio, model, type, ind,method, interval):
        self.param1 = prd
        self.param2 = z_score
        self.param3 = prd2
        self.step1  = step1
        self.step2 = step2
        self.step3 = step3
        self.df = df
        self.start = start
        self.end = end
        self.ratio = ratio
        self.model = model
        self.type = type
        self.ind = ind
        self.method = method
        self.interval_str = interval
        self.steps = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '10m' : 600,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1d': 86400,
            '3d': 259200,
            '1w': 604800,
        } ## in s

    def get_heatmap(self):
        full_result_dict = {}
        test_result_dict = {}
        train_result_dict = {}
        if self.interval_str not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            interval = self.steps[self.interval_str]
        KLINES_PER_YEAR = (365*24*3600) / interval
        df = self.df.loc[self.start:self.end]
        for i in np.arange(self.step1, self.param1, (self.param1 - self.step1) / 20):
            for j in np.arange(self.step2, self.param2, (self.param2 - self.step2) / 20):
                result = self.model(df, i, j, self.type,self.ind,self.method)
                train_result = result.loc[self.start:self.start + (self.end-self.start) * self.ratio]
                test_result = result.loc[self.start + (self.end-self.start) * self.ratio:]
                if train_result['return'].sum() != 0:
                    train_sharpe = qs.stats.sharpe(train_result['return'], periods=KLINES_PER_YEAR)
                else:
                    train_sharpe = 0

                if test_result['return'].sum() != 0:
                    test_sharpe = qs.stats.sharpe(test_result['return'], periods=KLINES_PER_YEAR)
                else:
                    test_sharpe = 0

                if test_result['return'].sum() != 0:
                    full_sharpe = qs.stats.sharpe(result['return'], periods=KLINES_PER_YEAR)
                else:
                    full_sharpe = 0

                train_result_dict.update({(i, j): train_sharpe})
                test_result_dict.update({(i, j): test_sharpe})
                full_result_dict.update({(i, j): full_sharpe})
        
        return train_result_dict, test_result_dict, full_result_dict
    
    def get_3D_heatmap(self):
        full_result_dict = {}
        test_result_dict = {}
        train_result_dict = {}
        KLINES_PER_YEAR = (365*24*3600) / self.interval
        df = self.df.loc[self.start:self.end]
        
        for i in np.arange(self.step1, self.param1, self.step1):
            for j in np.arange(self.step2, self.param2, self.step2):
                for k in np.arange(self.step3, self.param3, self.step3):
                    result = self.model(df, i, j, k, self.type, self.ind, self.method)  # Pass the third parameter k to your model
                    
                    train_result = result.loc[self.start:self.start + (self.end-self.start) * self.ratio]
                    test_result = result.loc[self.start + (self.end-self.start) * self.ratio:]
                    
                    if train_result['return'].sum() != 0:
                        train_sharpe = qs.stats.sharpe(train_result['return'], periods=KLINES_PER_YEAR)
                    else:
                        train_sharpe = 0

                    if test_result['return'].sum() != 0:
                        test_sharpe = qs.stats.sharpe(test_result['return'], periods=KLINES_PER_YEAR)
                    else:
                        test_sharpe = 0

                    if test_result['return'].sum() != 0:
                        full_sharpe = qs.stats.sharpe(result['return'], periods=KLINES_PER_YEAR)
                    else:
                        full_sharpe = 0

                    train_result_dict[(i, j, k)] = train_sharpe
                    test_result_dict[(i, j, k)] = test_sharpe
                    full_result_dict[(i, j, k)] = full_sharpe
            
        return train_result_dict, test_result_dict, full_result_dict
    

