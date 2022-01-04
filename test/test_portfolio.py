
import unittest
import os

import numpy as np

from finsim.portfolio import get_optimized_portfolio_on_sharpe_ratio, get_optimized_portfolio_on_mpt_costfunction
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


if __name__ == '__main__':
    unittest.main()
