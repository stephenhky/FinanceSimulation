
import unittest

import numpy as np

from finsim.portfolio import get_optimized_portfolio


class TestPortfolio(unittest.TestCase):
    def test_modern_portfolio_theory(self):
        selected_symbols = ['NVDA', 'AMZN', 'BAH', 'GD']

        startdate = '2010-01-01'
        enddate = '2020-03-31'

        optimized_portfolio = get_optimized_portfolio(0.0123, selected_symbols, 1000,
                                                      enddate, startdate, enddate,
                                                      minweight=0.2)

        summary = optimized_portfolio.summary
        
        self.assertAlmostEqual(summary['yield'], 0.29045820465394034)
        self.assertAlmostEqual(summary['volatility'], 0.24309206546829054)
        self.assertAlmostEqual(summary['sharpe_ratio'], 1.1442504473278414)
        np.testing.assert_array_almost_equal(
            summary['correlation'],
            np.array([[1.        , 0.38132613, 0.28049998, 0.42612949],
                      [0.38132613, 1.        , 0.19692159, 0.37066551],
                      [0.28049998, 0.19692159, 1.        , 0.34781397],
                      [0.42612949, 0.37066551, 0.34781397, 1.        ]]
                     )
        )

if __name__ == '__main__':
    unittest.main()
