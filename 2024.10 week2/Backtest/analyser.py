import pathlib
import logging
from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.loggers import custom_logger

__all__ = (
    # 'VBTPorfotlioBacktester',
    # 'VectorbtStrategyBacktester',
    'CustomStrategyBacktester',
    'backtest_logs',
)

backtest_logs = custom_logger('backtest_logs', to_file=True, filename='./logs/backtest_logs.log', mode='w+',
                              log_level=logging.DEBUG)


class CustomStrategyBacktester(object):
    def __init__(self, df: pd.DataFrame, result_dir: str, resolution: str, custom_msg: str = '', px_col: str = 'close',
                 log_returns: bool = False,
                 risk_free_rate: float = 0.0, transaction_fees=0.001, annual_days: int = 365,
                 bt_log: logging.Logger = backtest_logs) -> None:
        assert isinstance(df.index, pd.DatetimeIndex)
        self.df = df.copy()

        if log_returns is True:
            self.df['underlying_ret'] = np.log(self.df[px_col] / self.df[px_col].shift(1))

        self.df['underlying_ret'] = self.df[px_col].pct_change()
        self.result_dir = result_dir
        self.custom_msg = custom_msg
        self._time_factors = {
            '1d': annual_days,
            '8h': annual_days * 3,
            '4h': annual_days * 6,
            '1h': annual_days * 24,
            '30m': annual_days * 48,
            '15m': annual_days * 96,
            '10m' : annual_days * 144,
            '5m': annual_days * 288,
            '1m': annual_days * 1440
        }
        self.risk_free_rate = risk_free_rate
        self.bt_log = bt_log
        pathlib.Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        if self._time_factors.get(resolution) is None:
            raise ValueError(f'Invalid resolution: {resolution}')

        self.annual_days = annual_days
        self.multiplier = self._time_factors[resolution] / annual_days

        self.df['trade'] = self.df['pos'].fillna(0) - self.df['pos'].shift(1).fillna(0)
        self.df['trade_ref'] = self.df['trade'].abs().replace(2, 1).cumsum()
        self.df['tc'] = self.df['trade'].abs() * transaction_fees
        self.df['pnl'] = self.df['pos'] * self.df['underlying_ret'] - self.df['tc']
        # self.df['pnl'] = self.df['pos'].shift(1) * self.df['underlying_ret'] - self.df['tc']
        self.df['cumulative_pnl'] = self.df['pnl'].cumsum()
        self.df['drawdown'] = self.df['cumulative_pnl'] - self.df['cumulative_pnl'].cummax()

    @property
    def long_transactions(self):
        return self.df.query('pos > 0').groupby('trade_ref')['pnl'].sum().apply(np.exp) - 1

    @property
    def short_transactions(self):
        return self.df.query('pos < 0').groupby('trade_ref')['pnl'].sum().apply(np.exp) - 1

    @property
    def total_transactions(self):
        return pd.concat([self.long_transactions, self.short_transactions])

    @property
    def drawdown_series(self):
        return pd.concat([self.df['drawdown'].rename('drawdown'), self.cumulative_mdd.rename('mdd')], axis=1)

    @property
    def maximum_drawdown(self):
        return (-self.df['drawdown']).max()

    @property
    def current_drawdown(self):
        return - self.df['drawdown'][-2]

    @property
    def drawdown_periods(self):
        strategy_dd = self.df['drawdown']
        strategy_dd = strategy_dd[strategy_dd == 0]

        if len(strategy_dd) > 1:
            dd_periods = (strategy_dd.index[1:].to_pydatetime() - strategy_dd.index[
                                                                  :-1].to_pydatetime()).max()  # type: ignore
        else:
            dd_periods = None

        return dd_periods

    @property
    def cumulative_mdd(self):
        return self.df['drawdown'].cummin()

    @property
    def time_factors(self):
        return self._time_factors

    @property
    def long_trade_count(self):
        return len(self.long_transactions)

    @property
    def short_trade_count(self):
        return len(self.short_transactions)

    @property
    def total_trade_count(self):
        return len(self.total_transactions)

    @property
    def win_trade_count(self):
        return sum(self.total_transactions > 0)

    @property
    def loss_trade_count(self):
        return sum(self.total_transactions < 0)

    @property
    def max_loss_trade(self):
        return self.total_transactions.min()

    @property
    def max_win_trade(self):
        return self.total_transactions.max()

    @property
    def win_rate(self):
        return self.win_trade_count / self.total_trade_count

    @property
    def pnl(self):
        return self.df['pnl']

    @property
    def transaction_cost(self):
        return self.df['tc'].sum()

    @property
    def cumulative_return(self):
        return self.df['cumulative_pnl'].dropna().iloc[-1]

    @property
    def daily_return(self):
        return self.df['pnl'].resample('1D').sum()

    @property
    def sharpe_ratio(self):
        try:
            return self.df['pnl'].mean() / self.df['pnl'].std() * np.sqrt(self.multiplier * self.annual_days)
        except ZeroDivisionError:
            return np.nan

    @property
    def calmar_ratio(self):
        return self.daily_return.mean() * self.annual_days / self.maximum_drawdown

    @property
    def annualized_sharpe_ratio(self):
        return self.daily_return.mean() / self.daily_return.std() * np.sqrt(self.annual_days)

    @property
    def apy(self):
        return self.df['pnl'].mean() * self.multiplier * self.annual_days

    def long_short_return(self):
        total_return = self.df['cumulative_pnl'].rename('total_return')
        long_return = pd.Series(np.where(self.df['pos'] > 0, self.df['pnl'], 0), name='long_return',
                                index=self.df.index).cumsum()
        short_return = pd.Series(np.where(self.df['pos'] < 0, self.df['pnl'], 0), name='short_return',
                                 index=self.df.index).cumsum()
        returns = pd.concat([total_return, long_return, short_return], axis=1)
        return returns

    def strategy_statistics(self):
        return {
            'custom_msg': self.custom_msg,
            'start_time': self.df.index[0],
            'end_time': self.df.index[-1],
            'cumulative_return': self.cumulative_return,
            'daily_return': self.daily_return.mean(),
            'annual_return': self.daily_return.mean() * self.annual_days,
            'transaction_cost': self.transaction_cost,
            'sharpe_ratio': self.sharpe_ratio,
            'annualized_sharpe_ratio': self.annualized_sharpe_ratio,
            'apy': self.apy,
            'calmar_ratio': self.calmar_ratio,
            'maximum_drawdown': self.maximum_drawdown,
            'current_drawdown': self.current_drawdown,
            'total_trade_count': self.total_trade_count,
            'long_trade_count': self.long_trade_count,
            'short_trade_count': self.short_trade_count,
            'win_trade_count': self.win_trade_count,
            'loss_trade_count': self.loss_trade_count,
            'win_rate': self.win_rate,
            'max_win_trade': self.max_win_trade,
            'max_loss_trade': self.max_loss_trade,
            'drawdown_periods': self.drawdown_periods,
        }

    def strategy_statistics_table_export(self, file_name: str, fmt='csv'):
        stats_df = pd.DataFrame(self.strategy_statistics())

        if fmt == 'csv':
            stats_df.to_csv(f'{self.result_dir}/{file_name}.csv')
        elif fmt == 'parquet':
            stats_df.to_parquet(f'{self.result_dir}/{file_name}.parquet')
        else:
            raise ValueError(f'Invalid file format: {fmt}')

    def strategy_performance_plot(self, fig_name: Union[str, None] = None):
        all_returns = self.long_short_return()

        fig, axs = plt.subplots(4, sharex=True, figsize=(10, 15))

        # Plot PnLs
        all_returns.plot(ax=axs[0], y=['total_return', 'long_return', 'short_return'],
                         color=['deepskyblue', 'limegreen', 'tomato'], linewidth=2)
        axs[0].set_title('Cumulative PnLs')
        axs[0].legend(['Total Return', 'Long Return', 'Short Return'])

        # Plot PnLs
        self.df['pnl'].plot(ax=axs[1], color='chocolate', kind='line')
        axs[1].set_title('PnLs')

        # # Plot Position
        self.df['pos'].plot(ax=axs[2], color='violet', kind='line')
        axs[2].set_title('Position')

        # Plot Drawdown
        self.drawdown_series.plot(ax=axs[3], color=['darkorange', 'red'], kind='line')
        axs[3].set_title('Drawdown')

        if fig_name is not None:
            plt.savefig(f'{self.result_dir}/{fig_name}.png')
            self.bt_log.info('Result graph saved as {}'.format(f'{self.result_dir}/{fig_name}.png'))
        else:
            plt.show()

        return {'custom_msg': self.custom_msg, 'stats': self.strategy_statistics()}

    def backtest_statistics(self):
        return None

    def export_quantstat_report(self, filename: str, benchmark: str = 'BTC'):
        import quantstats as qs
        qs.extend_pandas()
        qs.reports.html(self.df['pnl'], benchmark=benchmark, download_filename=f'{filename}.html',
                        periods_per_year=self.annual_days * self.multiplier)  # type: ignore

        return

    def draw_histogram_for_stat_measure(self, stat, bins=50):
        plt.hist(self.df[stat], bins=50)
        plt.title(f'Histogram of {stat}')
        plt.show()