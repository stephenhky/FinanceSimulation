
import unittest
from math import exp, sqrt

import numpy as np

from finsim.simulation.stock import BlackScholesMertonStockPrices, HestonStockPrices, MertonJumpDiffusionStockPrices


def expected_bsm_stock(S0, r, sigma, t):
    expected_S = S0 * exp(r*t)
    variance_S = S0 * S0 * (exp(sigma*sigma*t)-1) * exp((2*r+sigma*sigma)*t)
    return expected_S, variance_S


class TestStockSimulations(unittest.TestCase):
    def backend_test_BlackScholesMertonStocks(self, S, r, sigma, T, dt, nbsimulations):
        bsm_simulator = BlackScholesMertonStockPrices(S, r, sigma)
        sarray = bsm_simulator.generate_time_series(T, dt, nbsimulations=nbsimulations)
        average_s = np.mean(sarray[:, -1])

        expected_s, variance_s = expected_bsm_stock(S, r, sigma, T)

        self.assertAlmostEqual(average_s, expected_s, delta=1.96*sqrt(variance_s))

    def test_BlackScholesMertonStocks(self):
        self.backend_test_BlackScholesMertonStocks(100, 0.04, 0.02, 100, 0.1, 10000)

        for S in np.random.lognormal(100, 10, size=3):
            for r in np.random.lognormal(0.04, 0.01, size=3):
                for sigma in np.random.lognormal(0.01, 0.0001, size=3):
                    self.backend_test_BlackScholesMertonStocks(S, r, sigma, 100, 0.1, 10000)

    def test_HestonStocks(self):
        h_simulator = HestonStockPrices(100, 0.01, 0.01, 0.01, 0.1, 0.0001, 0.2)
        h_stocks = [
            np.mean(h_simulator.generate_time_series(100, 0.1, nbsimulations=1000)[0], axis=0)
            for _ in range(100)
        ]

        self.assertAlmostEqual(
            np.mean([stock[100] for stock in h_stocks]),
            110.52,
            delta=1.96*0.11
        )
        self.assertAlmostEqual(
            np.mean([stock[250] for stock in h_stocks]),
            128.41,
            delta=1.96*0.23
        )
        self.assertAlmostEqual(
            np.mean([stock[500] for stock in h_stocks]),
            164.92,
            delta=1.96*0.38
        )
        self.assertAlmostEqual(
            np.mean([stock[750] for stock in h_stocks]),
            211.80,
            delta=1.96*0.56
        )
        self.assertAlmostEqual(
            np.mean([stock[999] for stock in h_stocks]),
            271.72,
            delta=1.96*0.80
        )

    def test_MertonJumpDiffusionStocks(self):
        mjd_simulator = MertonJumpDiffusionStockPrices(100, 0.01, 0.01, 0.01, 0.01, 0.001)
        mjd_stocks = [
            np.mean(mjd_simulator.generate_time_series(100, 0.1, nbsimulations=1000), axis=0)
            for _ in range(100)
        ]

        self.assertAlmostEqual(
            np.mean([stock[250] for stock in mjd_stocks]),
            197.97,
            delta=1.96*6.37
        )
        self.assertAlmostEqual(
            np.mean([stock[500] for stock in mjd_stocks]),
            393.35,
            delta=1.96*28.15
        )
        self.assertAlmostEqual(
            np.mean([stock[750] for stock in mjd_stocks]),
            773.20,
            delta=1.96*67.73
        )
        self.assertAlmostEqual(
            np.mean([stock[999] for stock in mjd_stocks]),
            1523.06,
            delta=1.96*169.68
        )


if __name__ == '__main__':
    unittest.main()
