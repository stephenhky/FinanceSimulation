
from datetime import datetime
from typing import Any

import requests
import pandas as pd
from pandas import DataFrame


class FinnHubStockReader:
    """A class to read stock data from FinnHub API."""
    
    def __init__(self, token: str):
        """Initialize the FinnHubStockReader with an API token.
        
        Args:
            token: The API token for FinnHub
        """
        self._token = token

    def get_all_US_symbols(self) -> list[dict[str, Any]]:
        """Get all US stock symbols from FinnHub.
        
        Returns:
            A list of dictionaries containing stock symbol information
        """
        request_url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={self._token}"
        r = requests.get(request_url)
        response = r.json()
        return response

    def get_stock_candlestick(self, symbol: str, startdate: str, enddate: str) -> DataFrame:
        """Get stock candlestick data for a given symbol and date range.
        
        Args:
            symbol: The stock symbol
            startdate: The start date in 'YYYY-MM-DD' format
            enddate: The end date in 'YYYY-MM-DD' format
            
        Returns:
            A DataFrame containing the candlestick data with columns:
            TimeStamp, High, Low, Open, Close, Volume
            
        Raises:
            Exception: If the API request fails
        """
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
