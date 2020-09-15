
import unittest
from math import exp, sqrt

import numpy as np

from finsim.simulation.stock import BlackScholesMertonStockPrices


def expected_stock(S0, r, sigma, t):
    expected_S = S0 * exp(r*t)
    variance_S = S0 * S0 * (exp(sigma*sigma*t)-1) * exp((2*r+sigma*sigma)*t)
    return expected_S, variance_S


class TestStockSimulations(unittest.TestCase):
    def backend_test_BlackScholesMertonStocks(self, S, r, sigma, T, dt, nbsimulations):
        bsm_simulator = BlackScholesMertonStockPrices(S, r, sigma)
        sarray = bsm_simulator.generate_time_series(T, dt, nbsimulations=nbsimulations)
        average_s = np.mean(sarray[:, -1])

        expected_s, variance_s = expected_stock(S, r, sigma, T)

        self.assertAlmostEqual(average_s, expected_s, delta=1.96*sqrt(variance_s))

    def test_test_BlackScholesMertonStocks(self):
        self.backend_test_BlackScholesMertonStocks(100, 0.04, 0.02, 100, 0.1, 10000)

        for S in np.random.lognormal(100, 10, size=3):
            for r in np.random.lognormal(0.04, 0.01, size=3):
                for sigma in np.random.lognormal(0.01, 0.0001, size=3):
                    self.backend_test_BlackScholesMertonStocks(S, r, sigma, 100, 0.1, 10000)


if __name__ == '__main__':
    unittest.main()
