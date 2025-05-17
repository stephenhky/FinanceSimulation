
from datetime import datetime
from typing import Any

import requests
import pandas as pd
from pandas import DataFrame


class FinnHubStockReader:
    def __init__(self, token: str):
        self._token = token

    def get_all_US_symbols(self) -> list[dict[str, Any]]:
        request_url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={self._token}"
        r = requests.get(request_url)
        response = r.json()
        return response

    def get_stock_candlestick(self, symbol: str, startdate: str, enddate: str) -> DataFrame:
        starttimestamp = int(datetime.strptime(startdate, '%Y-%m-%d').timestamp())
        endtimestamp = int(datetime.strptime(enddate, '%Y-%m-%d').timestamp())
        request_url = 'https://finnhub.io/api/v1/stock/candle?symbol={}&resolution=1&from={}&to={}&token={}'.format(symbol, starttimestamp, endtimestamp, self._token)
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
