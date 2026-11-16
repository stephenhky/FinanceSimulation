
import io
import unittest
from unittest.mock import patch
import os

import numpy as np
import pandas as pd

from finsim.portfolio import (
    get_optimized_portfolio_on_sharpe_ratio,
    get_optimized_portfolio_on_mpt_costfunction,
    get_optimized_portfolio_on_mpt_entropy_costfunction,
)
from finsim.portfolio.create import (
    get_exponential_timeweightdf,
    get_optimized_exponential_timeweighted_portfolio_on_mpt_entropy_costfunction,
)
from finsim.portfolio import OptimizedPortfolio, Portfolio


class TestPortfolio(unittest.TestCase):
    def test_modern_portfolio_sharpe_ratio(self):
        selected_symbols = ['NVDA', 'AMZN', 'BAH', 'GD']

        startdate = '2010-01-01'
        enddate = '2020-03-31'

        optimized_portfolio = get_optimized_portfolio_on_sharpe_ratio(0.0123, selected_symbols, 1000,
                                                                      enddate, startdate, enddate,
                                                                      minweight=0.2)

        summary = optimized_portfolio.summary
        
        self.assertAlmostEqual(summary['yield'], 0.290, places=3)
        self.assertAlmostEqual(summary['volatility'], 0.243, places=3)
        self.assertAlmostEqual(summary['sharpe_ratio'], 1.144, places=3)
        np.testing.assert_array_almost_equal(
            summary['correlation'],
            np.array([[1.        , 0.38132613, 0.28049998, 0.42612463],
                      [0.38132613, 1.        , 0.19692159, 0.37066065],
                      [0.28049998, 0.19692159, 1.        , 0.34782019],
                      [0.42612463, 0.37066065, 0.34782019, 1.        ]]
                     ),
            decimal=4
        )

        simplified_portfolio = optimized_portfolio.get_portfolio()
        self.assertTrue(isinstance(simplified_portfolio, Portfolio))
        self.assertFalse(isinstance(simplified_portfolio, OptimizedPortfolio))

    def test_modern_postfolio_costfunction(self):
        selected_symbols = ['NVDA', 'AMZN', 'BAH', 'GD']

        startdate = '2010-01-01'
        enddate = '2020-03-31'

        optimized_portfolio = get_optimized_portfolio_on_mpt_costfunction(0.0123, selected_symbols, 1000,
                                                                          enddate, startdate, enddate,
                                                                          0.1)

        summary = optimized_portfolio.summary

        self.assertAlmostEqual(summary['yield'], 0.354, places=3)
        self.assertAlmostEqual(summary['volatility'], 0.322, places=3)
        self.assertAlmostEqual(summary['mpt_costfunction'], 3.026, places=3)
        np.testing.assert_array_almost_equal(
            summary['correlation'],
            np.array([[1.        , 0.38132613, 0.28049998, 0.42612463],
                      [0.38132613, 1.        , 0.19692159, 0.37066065],
                      [0.28049998, 0.19692159, 1.        , 0.34782019],
                      [0.42612463, 0.37066065, 0.34782019, 1.        ]]
                     ),
            decimal=4
        )
        simplified_portfolio = optimized_portfolio.get_portfolio()
        self.assertTrue(isinstance(simplified_portfolio, Portfolio))
        self.assertFalse(isinstance(simplified_portfolio, OptimizedPortfolio))


        optimized_portfolio.save_to_json(open('portfolio.json', 'w'))
        reloaded_portfolio = Portfolio.load_from_json(open('portfolio.json', 'r'))
        self.assertAlmostEqual(
            optimized_portfolio.get_portfolio_value('2020-03-31'),
            reloaded_portfolio.get_portfolio_value('2020-03-31')
        )
        self.assertTrue(isinstance(reloaded_portfolio, Portfolio))
        self.assertFalse(isinstance(reloaded_portfolio, OptimizedPortfolio))
        os.remove('portfolio.json')

    def test_portfolio_arithmetic(self):
        portfolio1 = Portfolio({"NVDA": 1, "GOOG": 3, "JPM": 2})
        portfolio2 = Portfolio({"TSLA": 2, "NVDA": 3, "GOOG": 3, "CO": 2})

        portfolio3 = portfolio1 + portfolio2
        assert portfolio3.portfolio_symbols_nbshares["NVDA"] == 4
        assert portfolio3.portfolio_symbols_nbshares["GOOG"] == 6
        assert portfolio3.portfolio_symbols_nbshares["JPM"] == 2
        assert portfolio3.portfolio_symbols_nbshares["TSLA"] == 2
        assert portfolio3.portfolio_symbols_nbshares["CO"] == 2

        portfolio4 = portfolio1 - portfolio2
        assert portfolio4.portfolio_symbols_nbshares["NVDA"] == -2
        with self.assertRaises(KeyError):
            assert portfolio4.portfolio_symbols_nbshares["GOOG"] == 0
        assert portfolio4.portfolio_symbols_nbshares["JPM"] == 2
        assert portfolio4.portfolio_symbols_nbshares["TSLA"] == -2
        assert portfolio4.portfolio_symbols_nbshares["CO"] == -2

        assert portfolio1 != portfolio2

        portfolio5 = portfolio2 * 4.5
        assert portfolio5.portfolio_symbols_nbshares["TSLA"] == 9
        assert portfolio5.portfolio_symbols_nbshares["NVDA"] == 13.5
        assert portfolio5.portfolio_symbols_nbshares["GOOG"] == 13.5
        assert portfolio5.portfolio_symbols_nbshares["CO"] == 9

        portfolio6 = 4.5 * portfolio2
        assert portfolio6 == portfolio5

        portfolio5.roundoff_nbshares()
        self.assertEqual(portfolio5.portfolio_symbols_nbshares["NVDA"], 14)
        portfolio5.multiply(2.0)
        self.assertEqual(portfolio5.portfolio_symbols_nbshares["NVDA"], 28.0)

        reloaded_dict_port = Portfolio.load_from_dict(portfolio1.portfolio_symbols_nbshares)
        self.assertEqual(reloaded_dict_port, portfolio1)
        self.assertIsInstance(portfolio1.dumps_json(), str)

    @patch('finsim.portfolio.portfolio.get_yahoofinance_data')
    def test_get_portfolio_values_overtime_unequal_lengths(self, mock_yf):
        def fake_yf(symbol, startdate, enddate, cacheddir=None):
            if symbol == 'NVDA':
                dates = pd.date_range('2020-01-01', '2020-01-04', freq='D')
                return pd.DataFrame({'TimeStamp': dates, 'Close': [100.0, 110.0, 120.0, 130.0]})
            elif symbol == 'ABNB':
                dates = pd.date_range('2020-01-03', '2020-01-04', freq='D')
                return pd.DataFrame({'TimeStamp': dates, 'Close': [50.0, 60.0]})
            else:
                return pd.DataFrame({'TimeStamp': pd.Series(dtype='datetime64[ns]'), 'Close': pd.Series(dtype='float64')})

        mock_yf.side_effect = fake_yf
        port = Portfolio({'NVDA': 2, 'ABNB': 1, 'DEAD': 5})
        df = port.get_portfolio_values_overtime('2020-01-01', '2020-01-04')
        self.assertEqual(len(df), 4)
        self.assertEqual(list(df['value']), [200.0, 220.0, 290.0, 320.0])

    @patch('finsim.portfolio.portfolio.get_symbol_closing_price')
    @patch('finsim.portfolio.optimize.numerics.get_yahoofinance_data')
    @patch('finsim.portfolio.optimize.numerics.get_dividends_df')
    def test_entropy_and_timeweighted_optimized_portfolios(self, mock_div, mock_yf, mock_close):
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        def fake_yf(symbol, startdate, enddate, cacheddir=None):
            base = 100.0 if symbol == 'NVDA' else 200.0
            growth = 1.01 if symbol == 'NVDA' else 1.005
            sub_dates = dates if symbol == 'NVDA' else dates[2:]
            prices = [base * (growth ** i) for i in range(len(sub_dates))]
            return pd.DataFrame({'TimeStamp': sub_dates, 'Close': prices})

        mock_yf.side_effect = fake_yf
        mock_close.return_value = 100.0
        mock_div.return_value = pd.DataFrame({'TimeStamp': ['2020-01-05'], 'Dividends': [1.0]})

        opt_port = get_optimized_portfolio_on_mpt_entropy_costfunction(
            0.01, ['NVDA', 'AMZN'], 1000.0, '2020-01-10', '2020-01-01', '2020-01-10',
            lamb0=0.1, lamb1=0.05, V=10.0, include_dividends=True
        )
        self.assertEqual(opt_port.portfolio_symbols, ['NVDA', 'AMZN'])
        self.assertEqual(len(opt_port.weights), 2)
        self.assertIsNotNone(opt_port.portfolio_yield)
        self.assertIsNotNone(opt_port.volatility)
        self.assertEqual(opt_port.correlation_matrix.shape, (2, 2))
        self.assertIsInstance(opt_port.named_correlation_matrix, pd.DataFrame)
        self.assertIn('mpt_entropy_costfunction', opt_port.portfolio_summary)
        self.assertIsInstance(opt_port.dumps_json(), str)
        with self.assertRaises(NotImplementedError):
            OptimizedPortfolio.load_from_json(io.StringIO('{}'))
        with self.assertRaises(NotImplementedError):
            OptimizedPortfolio.load_from_dict({})

        tw_df = get_exponential_timeweightdf('2020-01-01', '2020-01-10', yearscale=1.0)
        self.assertEqual(len(tw_df), 10)
        tw_port = get_optimized_exponential_timeweighted_portfolio_on_mpt_entropy_costfunction(
            0.01, ['NVDA', 'AMZN'], 1000.0, '2020-01-10', '2020-01-01', '2020-01-10',
            yearscale=1.0, lamb0=0.1, lamb1=0.05, V=10.0, include_dividends=True
        )
        self.assertEqual(tw_port.portfolio_symbols, ['NVDA', 'AMZN'])


if __name__ == '__main__':
    unittest.main()
