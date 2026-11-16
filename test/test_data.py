
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from finsim.data.preader import (
    get_symbol_closing_price,
    get_yahoofinance_data,
    finding_missing_symbols_in_cache,
    generating_cached_yahoofinance_data,
    extract_batch_online_yahoofinance_data,
    get_dividends_df,
)
from finsim.data.finnhub import FinnHubStockReader
from finsim.data.quandl import QuandlReader


class TestDynamicPortfolio(unittest.TestCase):
    def test_get_voo_backtrack(self):
        self.assertAlmostEqual(
            get_symbol_closing_price('VOO', '2016-01-01', backtrack=True),
            186.92999267578125
        )

    def test_get_voo_nobacktrack(self):
        self.assertRaises(
            IndexError,
            get_symbol_closing_price,
            'VOO',
            '2016-01-01',
            backtrack=False
        )
        self.assertAlmostEqual(
            get_symbol_closing_price('VOO', '2015-12-31', backtrack=True),
            186.92999267578125
        )

    @patch('finsim.data.finnhub.requests.get')
    def test_finnhub_reader(self, mock_get):
        reader = FinnHubStockReader('test_token')

        mock_resp_symbols = MagicMock()
        mock_resp_symbols.json.return_value = [{'symbol': 'AAPL', 'mic': 'XNAS'}]
        mock_get.return_value = mock_resp_symbols
        symbols = reader.get_all_US_symbols()
        self.assertEqual(symbols, [{'symbol': 'AAPL', 'mic': 'XNAS'}])

        mock_resp_candle = MagicMock()
        mock_resp_candle.json.return_value = {
            's': 'ok',
            't': [1577836800, 1577923200],
            'h': [101.0, 102.0],
            'l': [99.0, 100.0],
            'o': [100.0, 101.0],
            'c': [100.5, 101.5],
            'v': [1000, 1200]
        }
        mock_get.return_value = mock_resp_candle
        candle_df = reader.get_stock_candlestick('AAPL', '2020-01-01', '2020-01-02')
        self.assertEqual(len(candle_df), 2)
        self.assertEqual(list(candle_df['Close']), [100.5, 101.5])

        mock_resp_err = MagicMock()
        mock_resp_err.json.return_value = {'s': 'no_data'}
        mock_get.return_value = mock_resp_err
        with self.assertRaises(Exception):
            reader.get_stock_candlestick('AAPL', '2020-01-01', '2020-01-02')

    @patch('finsim.data.quandl.quandl.get')
    def test_quandl_reader(self, mock_quandl_get):
        mock_quandl_get.return_value = pd.DataFrame({'Rate': [1.5, 1.6]})
        qreader = QuandlReader('quandl_token')

        self.assertEqual(len(qreader.get_treasury_real_longterm_rates()), 2)
        mock_quandl_get.assert_called_with("USTREASURY/REALLONGTERM", authtoken='quandl_token')

        self.assertEqual(len(qreader.get_treasury_real_yield_curve_rates()), 2)
        mock_quandl_get.assert_called_with("USTREASURY/REALYIELD", authtoken='quandl_token')

        self.assertEqual(len(qreader.get_treasury_yield_curve_rates()), 2)
        mock_quandl_get.assert_called_with("USTREASURY/YIELD", authtoken='quandl_token')

        self.assertEqual(len(qreader.get_treasury_longterm_rates()), 2)
        mock_quandl_get.assert_called_with("USTREASURY/LONGTERMRATES", authtoken='quandl_token')

    @patch('finsim.data.preader.extract_online_yahoofinance_data')
    def test_hdf5_caching_workflow(self, mock_extract):
        sample_df = pd.DataFrame({
            'TimeStamp': pd.to_datetime(['2020-01-02', '2020-01-03']),
            'High': [102.0, 103.0],
            'Low': [98.0, 99.0],
            'Open': [100.0, 101.0],
            'Close': [101.0, 102.0],
            'Adj Close': [101.0, 102.0],
            'Volume': [500, 600],
        })
        mock_extract.return_value = sample_df

        with tempfile.TemporaryDirectory() as tmpdir:
            # First call populates cache
            df1 = get_yahoofinance_data('AAPL', '2020-01-01', '2020-01-05', cacheddir=tmpdir)
            self.assertEqual(len(df1), 2)
            self.assertEqual(mock_extract.call_count, 1)

            # Second call within range reads from HDF5 cache without calling extract_online_yahoofinance_data
            df2 = get_yahoofinance_data('AAPL', '2020-01-02', '2020-01-03', cacheddir=tmpdir)
            self.assertEqual(len(df2), 2)
            self.assertEqual(mock_extract.call_count, 1)

            missing = finding_missing_symbols_in_cache(['AAPL', 'MSFT'], '2020-01-02', '2020-01-03', tmpdir)
            self.assertEqual(missing, ['MSFT'])

    @patch('finsim.data.preader.yf.Ticker')
    def test_get_dividends_df(self, mock_ticker_cls):
        get_dividends_df.cache_clear()
        mock_ticker = MagicMock()
        mock_ticker.dividends = pd.Series(
            [0.22, 0.23],
            index=pd.to_datetime(['2020-02-07', '2020-05-08']),
            name='Dividends'
        )
        mock_ticker_cls.return_value = mock_ticker
        df = get_dividends_df('AAPL')
        self.assertEqual(list(df.columns), ['TimeStamp', 'Dividends'])
        self.assertEqual(len(df), 2)


if __name__ == '__main__':
    unittest.main()
