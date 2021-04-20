
import unittest

import numpy as np

from finsim.estimate import risk


class TestRisk(unittest.TestCase):
    def test_nodownward_movement(self):
        ts = np.linspace(0, 10, 11)
        S = np.linspace(1.0, 1.8, 11)
        self.assertAlmostEqual(
            risk.python_estimate_downside_risk(ts, S, 0.0),
            0.0
        )
        self.assertAlmostEqual(
            risk.cython_estimate_downside_risk(ts, S, 0.0),
            0.0
        )
        self.assertAlmostEqual(
            risk.fortranrisk.fortran_estimate_downside_risk(ts, S, 0.0),
            0.0
        )

    def test_somedownward_movement(self):
        r_fake_array = np.array([0.1, 0.2, -0.1, -0.1, 0.3, 0.4])
        S_array = np.exp(np.cumsum(r_fake_array))
        ts = np.linspace(0, 5, 6)
        expected_downside_risk = np.sqrt(0.02/5)

        self.assertAlmostEqual(
            risk.python_estimate_downside_risk(ts, S_array, 0.0),
            expected_downside_risk
        )
        self.assertAlmostEqual(
            risk.cython_estimate_downside_risk(ts, S_array, 0.0),
            expected_downside_risk
        )
        self.assertAlmostEqual(
            risk.fortranrisk.fortran_estimate_downside_risk(ts, S_array, 0.0),
            expected_downside_risk
        )

    def test_noupward_movement(self):
        ts = np.linspace(0, 10, 11)
        S = np.linspace(1.0, 0.5, 11)

        self.assertAlmostEqual(
            risk.python_estimate_upside_risk(ts, S, 0.0),
            0.0
        )
        self.assertAlmostEqual(
            risk.cython_estimate_upside_risk(ts, S, 0.0),
            0.0
        )
        self.assertAlmostEqual(
            risk.fortranrisk.fortran_estimate_upside_risk(ts, S, 0.0),
            0.0
        )

    def test_someupward_movement(self):
        r_fake_array = np.array([-0.1, -0.2, 0.1, 0.1, -0.3, -0.4])
        S_array = np.exp(np.cumsum(r_fake_array))
        ts = np.linspace(0, 5, 6)
        expected_upside_risk = np.sqrt(0.02/5)

        self.assertAlmostEqual(
            risk.python_estimate_upside_risk(ts, S_array, 0.0),
            expected_upside_risk
        )
        self.assertAlmostEqual(
            risk.cython_estimate_upside_risk(ts, S_array, 0.0),
            expected_upside_risk
        )
        self.assertAlmostEqual(
            risk.fortranrisk.fortran_estimate_upside_risk(ts, S_array, 0.0),
            expected_upside_risk
        )

    def test_beta(self):
        timestamps = np.array(['2021-02-16T00:00:00.000000000', 
                               '2021-02-17T00:00:00.000000000', 
                               '2021-02-18T00:00:00.000000000', 
                               '2021-02-19T00:00:00.000000000'], 
                              dtype='datetime64[ns]')
        index_prices = 100*np.exp([0., 0.02, 0.02+0.01, 0.02+0.01-0.015])
        stock_prices = 10*np.exp([0., 0.04, 0.04+0.02, 0.04+0.02-0.03])
        beta = risk.estimate_beta(timestamps, stock_prices, index_prices)
        self.assertAlmostEqual(beta, 2.0)


if __name__ == '__main__':
    unittest.main()
