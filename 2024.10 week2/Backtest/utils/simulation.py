import config
import pandas as pd
import numpy as np
import warnings
import quantstats as qs
warnings.filterwarnings("ignore")

SPOT_COMMISSION = 3/10000 ## 3 bps point
FUTURE_COMMISSION = 10/10000 ## 10 bps point

def simulation(performance_df,Type, commission=6/10000):
    if (Type == 'Spot'):
        commission = SPOT_COMMISSION # from bps to decimal
    elif (Type == 'Futures'):
        commission = FUTURE_COMMISSION # 6bps to decimal
    elif (Type == 'Inverse'):
        commission = FUTURE_COMMISSION
    elif (Type == ''):
        commission = commission
    else:
        raise ValueError('market must be either "SPOT" or "FUTURES"')
    # does not consider fractional position
    performance_df['close'] = performance_df['close'].astype(float)
    performance_df['ROC'] = performance_df['close'].pct_change()
    if Type == 'Inverse':
        performance_df['Inverse_ROC'] = 1 - (performance_df['close'].shift(1) / performance_df['close'])
    else:
        performance_df['Inverse_ROC'] = 0
    performance_df['pos'] = performance_df['signal'].shift(1)  # position when signal was received (meaning, position is updated on the next row)
    performance_df['trade'] = performance_df['signal'].diff().shift(1)   # while signal reflects the new position, trade reflects the transaction made during close time
    performance_df['turnover'] = (abs(performance_df['trade']) * performance_df['close']).cumsum()  # turnover is the absolute value of the trade multiplied by the close price

    # condition = (performance_df['position'].shift() == performance_df['position']) # Calculate return with 1 time delay upon signal
    # performance_df['return'] = np.where(condition,
    # ((performance_df['position'] * performance_df['ROC'] + 1) *(1 - (abs(performance_df['trade']) * commission))) - 1,
    # (1 - (abs(performance_df['trade']) * commission)) - 1)
    performance_df['return_in_price'] = ((performance_df['pos'] * performance_df['ROC']+1)) -1 
    performance_df['return_in_inverse'] = ((performance_df['pos'] * performance_df['Inverse_ROC']+1)) -1
    performance_df['return'] = (1+performance_df['return_in_price']) * (1+performance_df['return_in_inverse']) * (1-(abs(performance_df['trade'])*commission))-1
    performance_df['Equity'] = 10000  # Set the initial equity value
    performance_df['Equity'] = performance_df['Equity'].shift(1) * (1 + performance_df['return']).cumprod()
    performance_df['unrealized_pnl'] = performance_df['return'].add(1).groupby((performance_df['pos'] != performance_df['pos'].shift()).cumsum()).cumprod().subtract(1)
    performance_df.loc[performance_df['trade'] != 0, 'realized_pnl'] = performance_df['unrealized_pnl']
    performance_df['cum_return'] = performance_df['return'].add(1).cumprod().sub(1)
    performance_df['cum_realized_pnl'] = performance_df['realized_pnl'].add(1).cumprod().sub(1)
    performance_df['long_return'] = performance_df.loc[performance_df['pos'] > 0, 'return']
    performance_df['short_return'] = performance_df.loc[performance_df['pos'] < 0, 'return']
    performance_df['cum_long_return'] = performance_df['long_return'].add(1).cumprod().sub(1)
    performance_df['cum_short_return'] = performance_df['short_return'].add(1).cumprod().sub(1)

    return performance_df


def simulation2(performance_df,Type):
    if (Type == 'Spot'):
        commission = 3 / 10000 # from bps to decimal
    elif (Type == 'Futures'):
        commission = 6 / 10000 # 6bps to decimal
    else:
        raise ValueError('market must be either "SPOT" or "FUTURES"')
        
    performance_df['return'] = ((performance_df['total_roc']+1)*(1-(abs(performance_df['trade'])*commission)))-1
    performance_df['Equity'] = 10000  # Set the initial equity value
    performance_df['Equity'] = performance_df['Equity'].shift(1) * (1 + performance_df['return']).cumprod()
    # performance_df['unrealized_pnl'] = performance_df['return'].add(1).groupby((performance_df['position'] != performance_df['position'].shift()).cumsum()).cumprod().subtract(1)
    # performance_df.loc[performance_df['trade'] != 0, 'realized_pnl'] = performance_df['unrealized_pnl']
    performance_df['cum_return'] = performance_df['return'].add(1).cumprod().sub(1)
    # performance_df['cum_realized_pnl'] = performance_df['realized_pnl'].add(1).cumprod().sub(1)
    # performance_df['long_return'] = performance_df.loc[performance_df['position'] > 0, 'return']
    # performance_df['short_return'] = performance_df.loc[performance_df['position'] < 0, 'return']
    # performance_df['cum_long_return'] = performance_df['long_return'].add(1).cumprod().sub(1)
    # performance_df['cum_short_return'] = performance_df['short_return'].add(1).cumprod().sub(1)

    return performance_df