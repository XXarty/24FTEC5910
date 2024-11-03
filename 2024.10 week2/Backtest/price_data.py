import json
import time
import warnings
from datetime import datetime, timezone
from typing import List, Union
import requests
import pandas as pd
# import okx.MarketData as MarketData
# import okx.PublicData as PublicData

__all__ = (
    'BinanceAPI',
    'BybitAPI',
    'CoinbaseAPI',
    'BitfitnexAPI',
    'UpbitAPI'
)

class BinanceAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.limit = 500
        self.instrument_base_url = {
            'spot': 'https://api.binance.com',
            'linear_derivatives': 'https://fapi.binance.com',
            'inverse_derivatives': 'https://dapi.binance.com',
        }
        self.steps = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
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
        }

    def get_kline(self, symbol: str, interval: str, instrument_type: str,
                  start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]

        if end_date is None:
            end_unixts = int(datetime.now().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        if instrument_type not in self.instrument_base_url.keys():
            raise ValueError('Invalid instrument_type')

        base_url = self.instrument_base_url[instrument_type]
        data = []

        if instrument_type == 'spot':
            kline_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                          'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            extend_url = '/api/v3/klines'
            limit = 1000
            step = step * 1000 * limit

        elif instrument_type == 'linear_derivatives':
            # https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
            extend_url = '/fapi/v1/klines'
            kline_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
                          'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
            limit = 1000
            step = step * 1000 * limit

        elif instrument_type == 'inverse_derivatives':
            # https://binance-docs.github.io/apidocs/delivery/en/#kline-candlestick-data
            extend_url = '/dapi/v1/klines'
            kline_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'base_asset_volume',
                          'number_of_trades', 'taker_buy_volume', 'taker_buy_base_asset_volume', 'ignore']
            limit = 200
            step = step * 1000 * limit


        else:
            raise ValueError('Invalid instrument_type')
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts * 1000, end_unixts * 1000, step):
            try:
                start_time = t
                end_time = t + step
                if end_time > int(datetime.now().timestamp()*1000):
                    end_time = int(datetime.now().timestamp()*1000)
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': limit
                }
                print(params)
                response = requests.get(base_url + extend_url, params=params)
                if response.json() == []:
                    continue
                data += response.json()
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {instrument_type}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['open_time'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['open_time'] = df['open_time'].astype(float)
        df.sort_values(by='open_time', ascending=True, inplace=True)
        df['date_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('date_time', inplace=True)
        return df
    
    def get_fundingrate(self, symbol: str, interval: str, instrument_type: str,
                  start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        interval = '8h'
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]

        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        if instrument_type not in self.instrument_base_url.keys():
            raise ValueError('Invalid instrument_type')

        base_url = self.instrument_base_url[instrument_type]
        data = []

        if instrument_type == 'linear_derivatives':
            # https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
            extend_url = '/fapi/v1/fundingRate'
            kline_cols = ['symbol', 'fundingTime', 'fundingRate', 'markPrice']
            limit = 500
            step = step * 1000 * limit

        elif instrument_type == 'inverse_derivatives':
            # https://binance-docs.github.io/apidocs/delivery/en/#kline-candlestick-data
            extend_url = '/dapi/v1/fundingRate'
            kline_cols = ['symbol', 'fundingTime', 'fundingRate', 'markPrice']
            limit = 500
            step = step * 1000 * limit
        else:
            raise ValueError('Invalid instrument_type')
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts * 1000, end_unixts * 1000, step):
            if t+step > int(datetime.utcnow().timestamp()*1000):
                t = int(datetime.utcnow().timestamp()*1000) - step 
            try:
                start_time = t
                end_time = t + step
                params = {
                    'symbol': symbol,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': limit # Maximum limit
                }
                response = requests.get(base_url + extend_url, params=params)
                if response.json() == []:
                    continue
                data += response.json()
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {instrument_type}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['fundingTime'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['date'] = df['fundingTime'].astype(float)
        df.sort_values(by='fundingTime', ascending=True, inplace=True)
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index('date', inplace=True)
        return df


from concurrent.futures import ThreadPoolExecutor, as_completed

def download_full_binance_data(token, path):
    bn = BinanceAPI('', '')
    intevals = ['1d', '4h', '1h', '15m', '5m', '1m']

    for interval in intevals:
        for ins in ['inverse_derivatives']:
            try:
                print(f'Downloading {token} {interval} {ins}')
                if ins == 'spot':
                    kline = bn.get_kline(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='spot', start_date='2020-06-01'
                    )
                elif ins == 'linear_derivatives':
                    kline = bn.get_kline(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='linear_derivatives', start_date='2020-06-01'
                    )
                else:
                    kline = bn.get_kline(
                        symbol=f'{token}USD_PERP', interval=interval, 
                        instrument_type='inverse_derivatives', start_date='2020-06-01'
                    )
                kline.to_parquet(fr'{path}/{token}_{interval}_binance_{ins}.parquet')
            except Exception as e:
                print(e.__str__())
                print(f'Error: {token} {interval} {ins}')
                time.sleep(2)
                continue

def download_full_binance_fundingrate(token, path):
    bn = BinanceAPI('', '')
    intevals = ['8h']

    for interval in intevals:
        for ins in ['inverse_derivatives']:
            try:
                print(f'Downloading {token} {interval} {ins}')
                if ins == 'linear_derivatives':
                    kline = bn.get_fundingrate(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='linear_derivatives', start_date='2020-06-01'
                    )
                elif ins == 'inverse_derivatives':
                    kline = bn.get_fundingrate(
                        symbol=f'{token}USD_PERP', interval=interval, 
                        instrument_type='inverse_derivatives', start_date='2020-06-01'
                    )
                else:
                    print('Invalid Instrument Type')
                kline.to_parquet(fr'{path}/{token}_{interval}_binance_{ins}_fundingrate.parquet')
            except Exception as e:
                print(e.__str__())
                print(f'Error: {token} {interval} {ins}')
                time.sleep(2)
                continue
    

class BybitAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.bybit.com'
        self.limit = 1000
        self.steps = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1d': 86400,
            '1M': 2678400,
            '1W': 604800,
        }
        self.interval_str = {
            '1m': '1',
            '3m': '3',
            '5m': '5',
            '15m': '15',
            '30m': '30',
            '1h': '60',
            '2h': '120',
            '4h': '240',
            '6h': '360',
            '8h': '480',
            '12h': '720',
            '1d': 'D',
            '1M': 'M',
            '1W': 'W',
        }
        
    def get_kline(self, symbol: str, interval: str, category: str,
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
            interval = self.interval_str[interval]
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        if category not in ['spot','inverse','linear']:
            raise ValueError('Invalid instrument_type')
        data = []
        extend_url = '/v5/market/kline'
        kline_cols = ['date','open','high','low','close','volume','turnover']
        step = step * 1000 * self.limit
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts * 1000, end_unixts * 1000, step):
            if t+step > int(datetime.utcnow().timestamp()*1000):
                t = int(datetime.utcnow().timestamp()*1000) - step 
            try:
                start_time = t
                end_time = t + step
                params = {
                    'category': category,
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': self.limit
                }
                response = requests.get(self.base_url + extend_url, params=params)
                if response.json() == []:
                    continue
                data = response.json()
                temp = pd.DataFrame(data["result"]["list"], columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {category}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['date'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['date'] = df['date'].astype(float)
        df.sort_values(by='date', ascending=True, inplace=True)
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index('date', inplace=True)
        return df
    
    def get_fundingrate(self, symbol: str, interval: str, category: str,
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        interval = '8h'
        step = 28800
        limit = 200
        extend_url = '/v5/market/funding/history'
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        if category not in ['inverse','linear']:
            raise ValueError('Invalid instrument_type')
        data = []
        kline_cols = ['symbol','fundingRate','fundingRateTimestamp']
        step = step * 1000 * limit
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts * 1000, end_unixts * 1000, step):
            if t+step > int(datetime.utcnow().timestamp()*1000):
                t = int(datetime.utcnow().timestamp()*1000) - step 
            try:
                start_time = t
                end_time = t + step
                params = {
                    'category': category,
                    'symbol': symbol,
                    'startTime': start_time,
                    'endTime': end_time,
                    'limit': limit
                }
                response = requests.get(self.base_url + extend_url, params=params)
                if response.json() == []:
                    continue
                data = response.json()
                temp = pd.DataFrame(data["result"]["list"], columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {category}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['fundingRateTimestamp'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['date'] = df['fundingRateTimestamp'].astype(float)
        df.sort_values(by='date', ascending=True, inplace=True)
        df['date'] = pd.to_datetime(df['date'], unit='ms')
        df.set_index('date', inplace=True)
        return df

from concurrent.futures import ThreadPoolExecutor, as_completed

def download_full_bybit_data(token,path):
    bn = BybitAPI('', '')
    intevals = ['1d', '4h', '1h', '15m', '5m', '1m']
    
    for interval in intevals:
        for ins in ['spot','inverse','linear']:
            try:
                print(f'Downloading {token} {interval} {ins}')
                if ins == 'spot':
                    kline = bn.get_kline(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='spot', start_date='2020-06-01'
                    )
                elif ins == 'linear':
                    kline = bn.get_kline(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='linear', start_date='2020-06-01'
                    )
                else:
                    kline = bn.get_kline(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='inverse', start_date='2020-06-01'
                    )
                kline.to_parquet(f'{path}/{token}_{interval}_bybit_{ins}.parquet')
            except Exception as e:
                print(e.__str__())
                print(f'Error: {token} {interval} {ins}')
                time.sleep(2)
                continue

def download_full_bybit_fundingrate(token,path):
    bn = BybitAPI('', '')
    intevals = ['8h']
    for interval in intevals:
        for ins in ['inverse','linear']:
            try:
                print(f'Downloading {token} {interval} {ins}')
                if ins == 'linear':
                    kline = bn.get_fundingrate(
                        symbol=f'{token}USDT', interval=interval, 
                        category='linear', start_date='2020-06-01'
                    )
                elif ins == 'inverse':
                    kline = bn.get_fundingrate(
                        symbol=f'{token}USDT', interval=interval, 
                        category='inverse', start_date='2020-06-01'
                    )
                else:
                    print('Invalid Instrument Type')
                kline.to_parquet(fr'{path}/{token}_{interval}_bybit_{ins}_fundingrate.parquet')
            except Exception as e:
                print(e.__str__())
                print(f'Error: {token} {interval} {ins}')
                time.sleep(2)
                continue

class CoinbaseAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.bybit.com/v5/market/kline'
        self.limit = 300
        self.steps = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '12h': 43200,
            '1d': 86400,
            '1M': 2678400,
            '1W': 604800,
        }
        self.interval_str = {
            '1m': '60',
            '3m': '180',
            '5m': '300',
            '15m': '900',
            '30m': '1800',
            '1h': '3600',
            '2h': '7200',
            '4h': '14400',
            '6h': '21600',
            '12h': '43200',
            '1d': '86400',
        }

    def get_kline(self, symbol: str, interval: str,
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
            interval = self.interval_str[interval]
        if end_date is None:
            end_unixts = int(datetime.now().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        base_url = f'https://api.exchange.coinbase.com/products/{symbol}/candles'
        data = []
        kline_cols = ['timestamp', 'low', 'high', 'open', 'close', 'volume']
        step = step * self.limit
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts, end_unixts, step):
            try:
                start_time = t
                end_time = t + step
                if start_time > int(datetime.now().timestamp()):
                    continue
                if end_time > int(datetime.now().timestamp()):
                    end_time = int(datetime.now().timestamp())

                params = {
                    'granularity': f'{interval}',
                    'start': str(start_time),
                    'end': str(end_time),
                }
                response = requests.get(base_url, params=params)
                d = eval(response.text)
                data.extend(d)
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval}')
                time.sleep(2)
                continue
        df = pd.DataFrame(data, columns=kline_cols)
        duplicates = df.duplicated(subset=['timestamp'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['timestamp'] = df['timestamp'].astype(float)
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        return df
    
def download_full_coinbase_data(token,path):
    bn = CoinbaseAPI('', '')
    token = ['BTC-USD','ETH-USD','LINK-USD','AVAX-USD','SOL-USD']
    intevals = ['1d', '6h', '1h', '15m', '5m', '1m']
    
    for interval in intevals:
        for ins in ['spot','inverse','linear']:
            try:
                print(f'Downloading {token} {interval} {ins}')
                if ins == 'spot':
                    kline = bn.get_kline(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='spot', start_date='2020-06-01'
                    )
                elif ins == 'linear':
                    kline = bn.get_kline(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='linear', start_date='2020-06-01'
                    )
                else:
                    kline = bn.get_kline(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='inverse', start_date='2020-06-01'
                    )
                kline.to_parquet(f'{path}/{token}_{interval}_bybit_{ins}.parquet')
            except Exception as e:
                print(e.__str__())
                print(f'Error: {token} {interval} {ins}')
                time.sleep(2)
                continue

class BitfitnexAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api-pub.bitfinex.com/v2'
        self.limit = 10000
        self.steps = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '3h': 10800,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1D': 86400,
            '1M': 2678400,
            '1W': 604800,
        }
    def get_kline(self, symbol: str, interval: str, category: str,
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        if category not in ['spot','inverse','linear']:
            raise ValueError('Invalid instrument_type')
        data = []
        candle = fr'candles/trade:{interval}:t{symbol}'
        extend_url = fr'/{candle}/hist'
        kline_cols = ['MTS','OPEN','CLOSE','HIGH','LOW','VOLUME']
        step = step * 1000 * self.limit
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts * 1000, end_unixts * 1000, step):
            if t+step > int(datetime.utcnow().timestamp()*1000):
                t = int(datetime.utcnow().timestamp()*1000) - step 
            try:
                start_time = t
                end_time = t + step
                params = {
                    'sort' : +1,
                    'start' : start_time, ## in ms
                    'limit' : 10000
                }
                # response = requests.get(temp_url, params=params)
                response = requests.get(self.base_url + extend_url, params=params)
                if response.json() == []:
                    continue
                data += response.json()
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {category}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['MTS'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['MTS'] = df['MTS'].astype(float)
        df.sort_values(by='MTS', ascending=True, inplace=True)
        df['MTS'] = pd.to_datetime(df['MTS'], unit='ms')
        df.set_index('MTS', inplace=True)
        return df
    
    def get_fundingrate(self, symbol: str, interval: str, category: str,
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        if category not in ['spot','inverse','linear']:
            raise ValueError('Invalid instrument_type')
        data = []
        candle = fr'candles/trade:{interval}:f{symbol}:p30'
        extend_url = fr'/{candle}/hist'
        kline_cols = ['MTS','OPEN','CLOSE','HIGH','LOW','VOLUME']
        step = step * 1000 * self.limit
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts * 1000, end_unixts * 1000, step):
            if t+step > int(datetime.utcnow().timestamp()*1000):
                t = int(datetime.utcnow().timestamp()*1000) - step 
            try:
                start_time = t
                end_time = t + step
                params = {
                    'sort' : +1,
                    'start' : start_time, ## in ms
                    'limit' : 10000
                }
                # response = requests.get(temp_url, params=params)
                response = requests.get(self.base_url + extend_url, params=params)
                if response.json() == []:
                    continue
                data += response.json()
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {category}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['MTS'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['MTS'] = df['MTS'].astype(float)
        df.sort_values(by='MTS', ascending=True, inplace=True)
        df['MTS'] = pd.to_datetime(df['MTS'], unit='ms')
        df.set_index('MTS', inplace=True)
        return df
def get_bitfitnex_symbollist():
    url = "https://api-pub.bitfinex.com/v2/tickers?symbols=ALL"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()

    df = pd.DataFrame(data)
    ## ignore all the test token, start with fTEST
    df = df[~df[0].str.contains('fTEST')]
    return df[0]


def download_full_bitfitnex_data(token,path):
    bn = BitfitnexAPI('', '')
    token = ['BTCUSD','ETHUSD','SOLUSD','AVAXUSD']
    intevals = ['1d', '4h', '1h', '15m', '5m', '1m']
    
    for interval in intevals:
        for ins in ['linear']:
            try:
                print(f'Downloading {token} {interval} {ins}')
                kline = bn.get_kline(
                    symbol=f'{token}USDT', interval=interval, 
                    instrument_type='spot', start_date='2020-06-01'
                )
                kline.to_parquet(f'{path}/{token}_{interval}_bitfitnex_{ins}.parquet')
            except Exception as e:
                print(e.__str__())
                print(f'Error: {token} {interval} {ins}')
                time.sleep(2)
                continue

def download_full_bitfitnex_fundingrate(token,path):
    bn = BitfitnexAPI('', '')
    tokens = ['BTCUSD','ETHUSD','SOLUSD','AVAXUSD']
    intervals = ['1d', '4h', '1h', '15m', '5m', '1m']
    for token in tokens:
            for interval in intervals:
                ins = 'linear'
                try:
                    print(f'Downloading {token} {interval} {ins}')
                    funding = bn.get_fundingrate(
                        symbol=f'{token}USDT', interval=interval, 
                        instrument_type='spot', start_date='2020-06-01'
                    )
                    funding.to_parquet(f'{path}/{token}_{interval}_bitfitnex_{ins}_fundingRate.parquet')
                except Exception as e:
                    print(e.__str__())
                    print(f'Error: {token} {interval} {ins}')
                    time.sleep(2)
                    continue

class UpbitAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.upbit.com/v1'
        self.steps = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '3h': 10800,
            '6h': 21600,
            '12h': 43200,
            '1D': 86400,
            '1M': 2678400,
            '1W': 604800,
        }
    def get_kline(self, symbol: str, interval: str, category: str,
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        if category not in ['minutes','days','weeks']:
            raise ValueError('Invalid Category')
        data = []
        if category == 'minutes':
            kline_cols = ['market', 'candle_date_time_utc', 'candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'timestamp',
                          'candle_acc_trade_price', 'candle_acc_trade_volume', 'unit']
            unit = int(self.steps[interval] / 60)
            extend_url = f'/candles/minutes/{unit}'
            limit = 200
            step = step * limit
        elif category == 'days':
            # https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
            extend_url = '/candles/days'
            kline_cols = ['market', 'candle_date_time_utc', 'candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'timestamp',
                          'candle_acc_trade_price', 'candle_acc_trade_volume', 'prev_closing_price','change_price','change_rate','converted_trade_price']
            limit = 200
            step = step * limit

        elif category == 'weeks':
            # https://binance-docs.github.io/apidocs/delivery/en/#kline-candlestick-data
            extend_url = '/candles/weeks'
            kline_cols = ['market', 'candle_date_time_utc', 'candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'timestamp',
                          'candle_acc_trade_price', 'candle_acc_trade_volume', 'first_day_of_period']
            limit = 200
            step = step * limit
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts , end_unixts , step):
            if t+step > int(datetime.utcnow().timestamp()):
                t = int(datetime.utcnow().timestamp()) - step 
            try:
                # start_time = t
                end_timestamp = t + step
                dt = datetime.fromtimestamp(end_timestamp)
                end_time = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                params = {
                    'market' : symbol,
                    'to' : end_time, ## in ms
                    'count' : limit
                }
                # response = requests.get(temp_url, params=params)
                response = requests.get(self.base_url + extend_url, params=params)
                if response.json() == []:
                    continue
                data += response.json()
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {category}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['candle_date_time_utc'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        # df['timestamp'] = df['timestamp'].astype(float)
        # df['timestamp'] = df['timestamp'] / 1000
        # df['timestamp'] = df['timestamp'].astype(int)
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['candle_date_time_utc'] = pd.to_datetime(df['candle_date_time_utc'])
        df.set_index('candle_date_time_utc', inplace=True)
        return df

class BitgetAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.bitget.com/'
        self.steps = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '6h': 21600,
            '12h': 43200,
            '1d': 86400,
            '3d': 259200,
            '1W': 604800,
        }
        self.granularities = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '4h': '4H',
            '6h': '6H',
            '12h': '12H',
            '1d': '1D',
            '3d': '3D',
            '1W': '1week',
        }
    def get_kline(self, symbol: str, interval: str, category: str,
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        if category not in ['spot','linear','inverse']:
            raise ValueError('Invalid Category')
        data = []
        if category == 'spot':
            kline_cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_asset_volume', 'quote_asset_volume', 'volume_in_USDT']
            extend_url = '/api/v2/spot/market/history-candles'
            granularity = self.granularities[interval]
            limit = 200
            step = min(90 * 86400 * 1000, 200*step*1000)
        elif category == 'linear':
            # https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
            extend_url = '/api/v2/mix/market/history-candles'
            kline_cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_asset_volume', 'quote_asset_volume']
            productType = 'USDT-FUTURES'
            granularity = self.granularities[interval]
            limit = 200
            step = min(90 * 86400 * 1000, 200*step*1000)
        elif category == 'inverse':
            extend_url = '/api/v2/mix/market/history-candles'
            kline_cols = ['timestamp', 'open', 'high', 'low', 'close', 'base_asset_volume', 'quote_asset_volume']
            productType = 'COIN-FUTURES'
            granularity = self.granularities[interval]
            limit = 200
            step = min(90 * 86400 * 1000, 200*step*1000)
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts * 1000 , end_unixts * 1000 , step):
            if t+step > int(datetime.utcnow().timestamp() * 1000):
                t = int(datetime.utcnow().timestamp() * 1000) - step 
            try:
                start_time = t
                end_time = t + step
                if category == 'spot':
                    params = {
                        'symbol' : symbol,
                        'granularity' : granularity,
                        'endTime' : end_time, ## in ms
                        'limit' : f'{limit}'
                    }
                else:
                    params = {
                        'symbol' : symbol,
                        'productType' : f'{productType}',
                        'granularity' : granularity,
                        'startTime' : start_time, ## in ms
                        'endTime' : end_time, ## in ms
                        'limit' : f'{limit}'
                    }
                # response = requests.get(temp_url, params=params)
                response = requests.get(self.base_url + extend_url, params=params)
                if response.json()['data'] == []:
                    continue
                data += response.json()['data']
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(1)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {category}')
                time.sleep(1)
                continue
        duplicates = df.duplicated(subset=['timestamp'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['timestamp'] = df['timestamp'].astype(float)
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_fundingrate(self, symbol: str, interval: str, category: str,
                  start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        if category not in ['linear','inverse']:
            raise ValueError('Invalid instrument_type')
        data = []
        extend_url = '/api/v2/mix/market/history-fund-rate'
        kline_cols = ['symbol', 'fundingTime', 'fundingRate']
        if category == 'linear':
            productType = 'USDT-FUTURES'
        elif category == 'inverse':
            productType = 'COIN-FUTURES'
        # limit = 500
        # step = step * 1000 * limit
        df = pd.DataFrame(data, columns=kline_cols)
        time_range = int((end_unixts - start_unixts) / (28800*100))
        for i in range(time_range):
            try:
                # start_time = t
                # end_time = t + step
                params = {
                        'symbol' : symbol,
                        'productType' : f'{productType}',
                        'pageSize' : '100',
                        'pageNo' : f'{i+1}'
                    }
                response = requests.get(self.base_url + extend_url, params=params)
                if response.json()['data'] == []:
                    continue
                data += response.json()['data']
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['fundingTime'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['fundingTime'] = df['fundingTime'].astype(float)
        df.sort_values(by='fundingTime', ascending=True, inplace=True)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df.set_index('fundingTime', inplace=True)
        return df
    
class KrakenAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.kraken.com/0/public/'
        self.steps = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '3h': 10800,
            '6h': 21600,
            '12h': 43200,
            '1d': 86400,
            '1W': 604800,
            '2W': 1209600
        }
        self.interval_str = {
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440,
            '1W': 10080,
            '2W': 21600
        }
    def get_kline(self, symbol: str, interval: str, 
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        data = []
        kline_cols = ['timestamp', 'open', 'high', 'low', 'close', 'VWAP', 'volume','count']
        extend_url = f'OHLC?pair={symbol}'
        limit = 720
        interval_str = self.interval_str[interval]
        step = step * limit
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts , end_unixts , step):
            if t+step > int(datetime.utcnow().timestamp()):
                t = int(datetime.utcnow().timestamp()) - step 
            try:
                start_time = t
                params = {
                    'pair' : symbol,
                    'interval' : interval_str, ## in ms
                    'since' : start_time
                }
                # response = requests.get(temp_url, params=params)
                response = requests.get(self.base_url + extend_url, params=params)
                print(response.json()['result'])
                # print(response.json()['result'][f'X{symbol[1:3]}Z{symbol[3:]}'].values())
                if response.json()['result'] == []:
                    continue
                data += response.json()['result']
                temp = pd.DataFrame(data['result']["XBTZUSD"], columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['timestamp'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        # df['timestamp'] = df['timestamp'].astype(float)
        # df['timestamp'] = df['timestamp'] / 1000
        # df['timestamp'] = df['timestamp'].astype(int)
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
class KucoinAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.limit = 500
        self.instrument_base_url = {
            'spot': 'https://api.kucoin.com',
            'linear': 'https://api-futures.kucoin.com',
        }
        self.steps = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '2h': 7200,
            '4h': 14400,
            '6h': 21600,
            '8h': 28800,
            '12h': 43200,
            '1d': 86400,
            '1W': 604800,
            '1M': 2678400
        }
        self.granularities = {
            '1m': '1min',
            '3m': '3min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1hour',
            '2h': '2hour',
            '4h': '4hour',
            '6h': '6hour',
            '12h': '12hour',
            '1d': '1day',
            '1W': '1week',
            '1M': '1month'
        }
    def get_kline(self, symbol: str, interval: str, instrument_type: str,
                  start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        if instrument_type not in self.instrument_base_url.keys():
            raise ValueError('Invalid instrument_type')

        base_url = self.instrument_base_url[instrument_type]
        data = []

        if instrument_type == 'spot':
            base_url = self.instrument_base_url[instrument_type]
            kline_cols = ['open_time', 'open', 'close', 'high', 'low', 'volume', 'turnover']
            extend_url = '/api/v1/market/candles'
            limit = 1500
            granularity = self.granularities[interval]
            step = step * limit

        elif instrument_type == 'linear':
            base_url = self.instrument_base_url[instrument_type]
            extend_url = '/api/v1/kline/query'
            kline_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
            limit = 200
            granularity = int(self.steps[interval]/60)
            step = step * limit
        else:
            raise ValueError('Invalid instrument_type')
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts , end_unixts , step):
            if t+step > int(datetime.utcnow().timestamp()):
                t = int(datetime.utcnow().timestamp()) - step 
            try:
                start_time = t
                end_time = t + step
                if instrument_type == 'spot':
                    params = {
                        'symbol': symbol,
                        'startAt': start_time,
                        'endAt': end_time,
                        'type': granularity
                    }
                elif instrument_type == 'linear':
                    params = {
                        'symbol': symbol,
                        'granularity': granularity,
                        'from': start_time * 1000,
                        'to': end_time * 1000
                    }
                response = requests.get(base_url + extend_url, params=params)
                if response.json()['data'] == []:
                    continue
                data += response.json()['data']
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {instrument_type}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['open_time'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['open_time'] = df['open_time'].astype(float)
        df.sort_values(by='open_time', ascending=True, inplace=True)
        if instrument_type == 'spot':
            df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
        else:
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        return df
    
    def get_fundingrate(self, symbol: str, interval: str, instrument_type: str,
                    start_date: str = '2020-06-01', end_date: Union[str, None] = None):
            interval = '8h'
            if interval not in self.steps.keys():
                raise ValueError('Invalid interval')
            else:
                step = self.steps[interval]

            if end_date is None:
                end_unixts = int(datetime.utcnow().timestamp())
            else:
                end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

            start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

            if instrument_type not in self.instrument_base_url.keys():
                raise ValueError('Invalid instrument_type')

            base_url = self.instrument_base_url[instrument_type]
            data = []

            if instrument_type == 'linear':
                # https://binance-docs.github.io/apidocs/futures/en/#kline-candlestick-data
                extend_url = '/api/v1/contract/funding-rates'
                kline_cols = ['symbol','fundingRate', 'timepoint']
                limit = 100
                step = step * 1000 * limit

            else:
                raise ValueError('Invalid instrument_type')
            df = pd.DataFrame(data, columns=kline_cols)
            for t in range(start_unixts * 1000, end_unixts * 1000, step):
                if t+step > int(datetime.utcnow().timestamp()*1000):
                    t = int(datetime.utcnow().timestamp()*1000) - step 
                try:
                    start_time = t
                    end_time = t + step
                    params = {
                        'symbol': symbol,
                        'from': start_time,
                        'to': end_time,
                    }
                    response = requests.get(base_url + extend_url, params=params)
                    if response.json()['data'] == []:
                        continue
                    data += response.json()['data']
                    temp = pd.DataFrame(data, columns=kline_cols)
                    df = pd.concat([df, temp])
                    time.sleep(2)
                except Exception as e:
                    print(e.__str__())
                    print(f'Error: {symbol} {interval} {instrument_type}')
                    time.sleep(2)
                    continue
            duplicates = df.duplicated(subset=['timepoint'])
            df = df[~duplicates]
            df = df.reset_index(drop=True)
            df['timepoint'] = df['timepoint'].astype(float)
            df.sort_values(by='timepoint', ascending=True, inplace=True)
            df['timepoint'] = pd.to_datetime(df['timepoint'], unit='ms')
            df.set_index('timepoint', inplace=True)
            return df
    

class GateioAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.gateio.ws/api/v4'
        self.steps = {
            '10s': 10,
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '8h': 28800,
            '1d': 86400,
            '7d': 604800,
            '30d': 2592000
        }
    def get_kline(self, symbol: str, interval: str, category: str,
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        if category not in ['spot','linear','inverse']:
            raise ValueError('Invalid Category')
        data = []
        if category == 'spot':
            kline_cols = ['timestamp', 'quote_asset_volume', 'close', 'high', 'low', 'open', 'base_asset_volume', 'candle_closed']
            extend_url = '/spot/candlesticks'
            limit = 900
            step = step * limit
        elif category == 'linear':
            settle = 'usdt'
            extend_url = f'/futures/{settle}/candlesticks'
            kline_cols = ['timestamp', 'volume', 'close', 'high', 'low', 'open', 'quote_asset_volume']
            limit = 1900
            step = step * limit
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts , end_unixts , step):
            if t+step > int(datetime.utcnow().timestamp()):
                t = int(datetime.utcnow().timestamp()) - step 
            try:
                start_time = t
                end_time = t + step
                if category == 'spot':
                    params = {
                        'currency_pair' : symbol,
                        'limit' : '1000',
                        'from' : start_time, ## in s
                        'to' : end_time,
                        'interval': interval
                    }
                else:
                    params = {
                        'contract' : symbol,
                        'from' : start_time,
                        'to' : end_time,
                        'limit' : '1000', ## in ms
                        'interval' : interval ## in ms
                    }
                # response = requests.get(temp_url, params=params)
                response = requests.get(self.base_url + extend_url, params=params)
                print(response.json())
                if response.json() == []:
                    continue
                data += response.json()
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(1)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {category}')
                time.sleep(1)
                continue
        duplicates = df.duplicated(subset=['timestamp'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['timestamp'] = df['timestamp'].astype(float)
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_fundingrate(self, symbol: str, interval: str, category: str,
                  start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        if category not in ['linear','inverse']:
            raise ValueError('Invalid instrument_type')
        data = []
        extend_url = '/futures/usdt/funding_rate'
        kline_cols = ['t', 'r']
        # limit = 500
        # step = step * 1000 * limit
        df = pd.DataFrame(data, columns=kline_cols)
        time_range = int((end_unixts - start_unixts) / (28800*100))
        for i in range(time_range):
            try:
                # start_time = t
                # end_time = t + step
                params = {
                        'contract' : symbol,
                        'limit' : 1000,
                    }
                response = requests.get(self.base_url + extend_url, params=params)
                # print(response.json())
                data += response.json()
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(2)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval}')
                time.sleep(2)
                continue
        duplicates = df.duplicated(subset=['t'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['t'] = df['t'].astype(float)
        df.sort_values(by='t', ascending=True, inplace=True)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df.set_index('t', inplace=True)
        return df
    
    def get_fundingstats(self, symbol: str, interval: str, category: str,
                start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        if category not in ['spot','linear','inverse']:
            raise ValueError('Invalid Category')
        data = []
        if category == 'linear':
            settle = 'usdt'
            extend_url = f'/futures/{settle}/contract_stats'
            # kline_cols = ['time', 'lsr_taker', 'lsr_account', 'long_liq_size', 'short_liq_size',
            #                'open_interest', 'mark_price', 'top_lsr_size', 'top_lsr_account', 'open_interest_usd']
            kline_cols = ['time', 'lsr_taker', 'lsr_account', 'long_liq_size', 'short_liq_size',
                    'open_interest', 'short_liq_usd', 'mark_price', 'top_lsr_size',
                    'short_liq_amount', 'long_liq_amount', 'open_interest_usd',
                    'top_lsr_account', 'long_liq_usd']
            limit = 100
            step = step * limit
        else:
            raise ValueError('Invalid Category')
        df = pd.DataFrame(data, columns=kline_cols)
        for t in range(start_unixts , end_unixts , step):
            if t+step > int(datetime.utcnow().timestamp()):
                t = int(datetime.utcnow().timestamp()) - step 
            try:
                start_time = t
                end_time = t + step
                params = {
                    'contract' : symbol,
                    'from' : start_time, ## in s
                    'interval': interval,
                    'limit' : 100
                }
                # response = requests.get(temp_url, params=params)
                response = requests.get(self.base_url + extend_url, params=params)
                if response.json() == []:
                    continue
                data += response.json()
                temp = pd.DataFrame(data, columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(1)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {category}')
                time.sleep(1)
                continue
        duplicates = df.duplicated(subset=['time'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['time'] = df['time'].astype(float)
        df.sort_values(by='time', ascending=True, inplace=True)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df


class OKXAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.limit = 100
        self.steps = {
            '1m': 60,
            '3m': 180,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1H': 3600,
            '2H': 7200,
            '4H': 14400,
            '6H': 21600,
            '8H': 28800,
            '12H': 43200,
            '1D': 86400,
            '3D': 259200,
            '1W': 604800,
        }

    def get_kline(self, symbol: str, interval: str, instrument_type: str,
                  start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]
            interval = f'{interval}utc'
        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    
        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        data = []
        if instrument_type == 'spot':
            kline_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume_base',
                          'volume_ccy_quote', 'volume_ccy_quote', 'completed']
            limit = 100
            step = step * 1000 * limit

        elif instrument_type == 'linear_derivatives':
            kline_cols = ['timestamp', 'open', 'high', 'low', 'close',  'volume_contract',
                          'volume_ccy_base', 'volume_ccy_quote', 'completed']
            limit = 100
            step = step * 1000 * limit

        elif instrument_type == 'inverse_derivatives':
            kline_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume_contract',
                          'volume_ccy_base', 'volume_ccy_quote', 'completed']
            limit = 100
            step = step * 1000 * limit
        else:
            raise ValueError('Invalid instrument_type')
        flag = '0'
        marketDataAPI = MarketData.MarketAPI(flag=flag)
        df = pd.DataFrame(data, columns=kline_cols)
        result = []
        for t in range(start_unixts * 1000 , end_unixts * 1000 , step):
            if t+step > int(datetime.utcnow().timestamp() * 1000):
                t = int(datetime.utcnow().timestamp() * 1000) - step 
            try:
                start_time = t
                end_time = t + step
                batch_result = marketDataAPI.get_candlesticks(
                    instId =f'{symbol}',
                    before = f'{start_time}',
                    after = f'{end_time}',
                    bar = interval,
                    limit=100
                )
                print(batch_result['data'])
                temp = pd.DataFrame(batch_result['data'], columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(1)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {instrument_type}')
                time.sleep(1)
                continue
        duplicates = df.duplicated(subset=['timestamp'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['timestamp'] = df['timestamp'].astype(float)
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        df['date_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date_time', inplace=True)
        return df
    
    def get_fundingrate(self, symbol: str, interval: str, instrument_type: str,
                  start_date: str = '2020-06-01', end_date: Union[str, None] = None):
        if interval not in self.steps.keys():
            raise ValueError('Invalid interval')
        else:
            step = self.steps[interval]

        if end_date is None:
            end_unixts = int(datetime.utcnow().timestamp())
        else:
            end_unixts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        start_unixts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())

        data = []
        if instrument_type == 'linear_derivatives':
            kline_cols = ['timestamp', 'open', 'high', 'low', 'close',  'volume_contract',
                          'volume_ccy_base', 'volume_ccy_quote', 'completed']
            limit = 100
            step = step * 1000 * limit

        elif instrument_type == 'inverse_derivatives':
            kline_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume_contract',
                          'volume_ccy_base', 'volume_ccy_quote', 'completed']
            limit = 100
            step = step * 1000 * limit
        else:
            raise ValueError('Invalid instrument_type')
        flag = '0'
        publicDataAPI = PublicData.PublicAPI(flag=flag)
        df = pd.DataFrame(data, columns=kline_cols)
        result = []
        for t in range(start_unixts * 1000 , end_unixts * 1000 , step):
            if t+step > int(datetime.utcnow().timestamp() * 1000):
                t = int(datetime.utcnow().timestamp() * 1000) - step 
            try:
                start_time = t
                end_time = t + step
                batch_result = publicDataAPI.get_funding_rate_history(
                    instId =f'{symbol}',
                    before = f'{start_time}',
                    after = f'{end_time}',
                    limit=100
                )
                print(batch_result['data'])
                temp = pd.DataFrame(batch_result['data'], columns=kline_cols)
                df = pd.concat([df, temp])
                time.sleep(1)
            except Exception as e:
                print(e.__str__())
                print(f'Error: {symbol} {interval} {instrument_type}')
                time.sleep(1)
                continue
        duplicates = df.duplicated(subset=['timestamp'])
        df = df[~duplicates]
        df = df.reset_index(drop=True)
        df['timestamp'] = df['timestamp'].astype(float)
        df.sort_values(by='timestamp', ascending=True, inplace=True)
        df['date_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date_time', inplace=True)
        return df
 
if __name__ == '__main__':
    tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'LINK', 'AVAX']

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_full_binance_data, t) for t in tokens]

