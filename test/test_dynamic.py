
import io
import unittest
from unittest.mock import patch
import logging
import pandas as pd

from finsim.portfolio import DynamicPortfolio, DynamicPortfolioWithDividends
from finsim.portfolio.helper import InsufficientSharesException

logging.basicConfig(level=logging.DEBUG)


class TestDynamicPortfolio(unittest.TestCase):
    def setUp(self):
        self.dynport = DynamicPortfolio({'NVDA': 1}, '2020-01-01')
        self.dynport.trade('2020-02-01', buy_stocks={'NVDA': 2, 'VOO': 1})
        self.dynport.trade('2020-03-01', buy_stocks={'LMT': 1}, sell_stocks={'NVDA': 3})
        self.dynport.trade('2020-04-01', buy_stocks={'COF': 2}, sell_stocks={'LMT': 2})

    def test_orders(self):
        self.assertTrue(self.dynport.is_sorted())
        self.assertEqual(self.dynport.find_cursor_for_date('2018-12-15'), -1)
        self.assertEqual(self.dynport.find_cursor_for_date('2020-01-06'), 0)
        self.assertEqual(self.dynport.find_cursor_for_date('2020-02-14'), 1)
        self.assertEqual(self.dynport.find_cursor_for_date('2020-03-22'), 2)
        self.assertEqual(self.dynport.find_cursor_for_date('2020-04-28'), 3)

    def test_sort_and_is_sorted(self):
        port = DynamicPortfolio({'NVDA': 1}, '2020-02-01')
        port.timeseries.append({
            'date': '2020-01-01',
            'portfolio': port.timeseries[0]['portfolio']
        })
        self.assertFalse(port.is_sorted())
        port.sort_time_series()
        self.assertTrue(port.is_sorted())

    def test_timevalue(self):
        self.assertAlmostEqual(
            self.dynport.get_portfolio_value('2020-03-02'),
            self.dynport.timeseries[2]['portfolio'].get_portfolio_value('2020-03-02')
        )

    def test_negative_shares(self):
        self.assertLess(self.dynport.timeseries[3]['portfolio'].symbols_nbshares['LMT'], 0)

    def test_trading_notcheck(self):
        port1 = DynamicPortfolio({'NVDA': 2, 'MSFT': 3}, '2020-01-01')
        port1.trade('2020-01-15', buy_stocks={'LMT': 3, 'NVDA': 1}, sell_stocks={'MSFT': 10})
        self.assertEqual(port1.symbols_nbshares['NVDA'], 3)
        self.assertEqual(port1.symbols_nbshares['LMT'], 3)
        self.assertEqual(port1.symbols_nbshares['MSFT'], -7)

    def test_trading_overlapping_buy_and_sell(self):
        port = DynamicPortfolio({'NVDA': 5, 'MSFT': 5, 'AAPL': 5}, '2020-01-01')
        port.trade(
            '2020-01-15',
            buy_stocks={'NVDA': 4, 'MSFT': 2, 'AAPL': 3},
            sell_stocks={'NVDA': 1, 'MSFT': 5, 'AAPL': 3}
        )
        self.assertEqual(port.symbols_nbshares['NVDA'], 8)
        self.assertEqual(port.symbols_nbshares['MSFT'], 2)
        self.assertEqual(port.symbols_nbshares['AAPL'], 5)

    def test_trading_check_available_fund(self):
        port2 = DynamicPortfolio({'NVDA': 2, 'MSFT': 3}, '2020-01-01')
        port2.trade('2020-01-15', buy_stocks={'LMT': 3, 'NVDA': 1}, sell_stocks={'MSFT': 10},
                    check_valid=True)
        self.assertEqual(port2.symbols_nbshares['NVDA'], 3)
        self.assertEqual(port2.symbols_nbshares['LMT'], 3)
        self.assertEqual(port2.symbols_nbshares['MSFT'], 0)

    def test_trading_raise_exception(self):
        port3 = DynamicPortfolio({'NVDA': 2, 'MSFT': 3}, '2020-01-01')
        with self.assertRaises(InsufficientSharesException):
            port3.trade('2020-01-15', buy_stocks={'LMT': 3, 'NVDA': 1}, sell_stocks={'MSFT': 10},
                        check_valid=True, raise_insufficient_stock_error=True)

    def test_serialization_json_and_dict(self):
        dynport_dict = self.dynport.generate_dynamic_portfolio_dict()
        self.assertEqual(dynport_dict['name'], 'DynamicPortfolio')
        self.assertEqual(len(dynport_dict['timeseries']), 4)

        reloaded_from_dict = DynamicPortfolio.load_from_dict(dynport_dict)
        self.assertEqual(reloaded_from_dict.current_date, self.dynport.current_date)
        self.assertEqual(reloaded_from_dict.symbols_nbshares, self.dynport.symbols_nbshares)

        buf = io.StringIO()
        self.dynport.save_to_json(buf)
        self.assertEqual(buf.getvalue(), self.dynport.dumps_json())
        buf.seek(0)
        reloaded_from_json = DynamicPortfolio.load_from_json(buf)
        self.assertEqual(reloaded_from_json.symbols_nbshares, self.dynport.symbols_nbshares)

    @patch('finsim.portfolio.portfolio.get_yahoofinance_data')
    def test_get_portfolio_values_overtime_and_dividends(self, mock_yf):
        def fake_yf(symbol, startdate, enddate, cacheddir=None):
            dates = pd.date_range(startdate, enddate, freq='D')
            price = 100.0 if symbol == 'NVDA' else 50.0
            return pd.DataFrame({
                'TimeStamp': dates,
                'Close': [price] * len(dates)
            })

        mock_yf.side_effect = fake_yf

        port = DynamicPortfolio({'NVDA': 2}, '2020-01-01')
        port.trade('2020-01-03', buy_stocks={'MSFT': 1})
        df = port.get_portfolio_values_overtime('2020-01-01', '2020-01-04')
        self.assertEqual(list(df['TimeStamp']), ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04'])
        self.assertEqual(list(df['value']), [200.0, 200.0, 250.0, 250.0])

        with patch('finsim.portfolio.dynamic.get_dividends_df') as mock_div:
            def fake_div(symbol):
                if symbol == 'NVDA':
                    return pd.DataFrame({'TimeStamp': ['2020-01-02'], 'Dividends': [1.5]})
                return pd.DataFrame({'TimeStamp': [], 'Dividends': []})

            mock_div.side_effect = fake_div
            div_port = DynamicPortfolioWithDividends({'NVDA': 2}, '2020-01-01', cash=10.0)
            div_port.trade('2020-01-03', buy_stocks={'MSFT': 1})
            div_df = div_port.get_portfolio_values_overtime('2020-01-01', '2020-01-04')
            self.assertEqual(list(div_df['cash']), [10.0, 13.0, 13.0, 13.0])
            self.assertEqual(list(div_df['value']), [210.0, 213.0, 263.0, 263.0])


if __name__ == '__main__':
    unittest.main()
