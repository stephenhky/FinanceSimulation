
from datetime import datetime

import requests
import pandas as pd


class FinnHubStockReader:
    def __init__(self, token):
        self.token = token

    def get_all_US_symbols(self):
        request_url = 'https://finnhub.io/api/v1/stock/symbol?exchange=US&token={}'.format(self.token)
        r = requests.get(request_url)
        response = r.json()
        return response

    # allsymbols_df = pd.DataFrame(finnhubReader.get_all_US_symbols())

    def get_stock_candlestick(self, symbol, startdate, enddate):
        starttimestamp = int(datetime.strptime(startdate, '%Y-%m-%d').timestamp())
        endtimestamp = int(datetime.strptime(enddate, '%Y-%m-%d').timestamp())
        request_url = 'https://finnhub.io/api/v1/stock/candle?symbol={}&resolution=1&from={}&to={}&token={}'.format(symbol, starttimestamp, endtimestamp, self.token)
        r = requests.get(request_url)
        response = r.json()
        if response['s'] == 'ok':
            df = pd.DataFrame({
                'TimeStamp': [pd.Timestamp(ts, unit='s') for ts in response['t']],
                'High': response['h'],
                'Low': response['l'],
                'Open': response['o'],
                'Close': response['c'],
                'Volume': response['v']
            })
            return df
        else:
            raise Exception('Error: {}'.format(response['s']))
