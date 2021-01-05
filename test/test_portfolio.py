
import unittest

import numpy as np

from finsim.portfolio import get_optimized_portfolio_on_sharpe_ratio, get_optimized_portfolio_on_mpt_costfunction


class TestPortfolio(unittest.TestCase):
    def test_modern_portfolio_sharpe_ratio(self):
        selected_symbols = ['NVDA', 'AMZN', 'BAH', 'GD']

        startdate = '2010-01-01'
        enddate = '2020-03-31'

        optimized_portfolio = get_optimized_portfolio_on_sharpe_ratio(0.0123, selected_symbols, 1000,
                                                                      enddate, startdate, enddate,
                                                                      minweight=0.2)

        summary = optimized_portfolio.summary
        
        self.assertAlmostEqual(summary['yield'], 0.29045820465394034)
        self.assertAlmostEqual(summary['volatility'], 0.24309206546829054)
        self.assertAlmostEqual(summary['sharpe_ratio'], 1.1442504473278414)
        np.testing.assert_array_almost_equal(
            summary['correlation'],
            np.array([[1., 0.38132613, 0.28049998, 0.42612949],
                      [0.38132613, 1., 0.19692159, 0.37066551],
                      [0.28049998, 0.19692159, 1., 0.34781397],
                      [0.42612949, 0.37066551, 0.34781397, 1.]]
                     )
        )

    def test_modern_postfolio_costfunction(self):
        selected_symbols = ['NVDA', 'AMZN', 'BAH', 'GD']

        startdate = '2010-01-01'
        enddate = '2020-03-31'

        optimized_portfolio = get_optimized_portfolio_on_mpt_costfunction(0.0123, selected_symbols, 1000,
                                                                          enddate, startdate, enddate,
                                                                          0.1)

        summary = optimized_portfolio.summary

        self.assertAlmostEqual(summary['yield'], 0.3115743402331405)
        self.assertAlmostEqual(summary['volatility'], 0.2620352447045725)
        self.assertAlmostEqual(summary['mpt_costfunction'], 6.737315202603957)
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
