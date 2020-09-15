
import unittest
from math import exp, sqrt

import numpy as np

from finsim.simulation.stock import BlackScholesMertonStockPrices


def expected_stock(S0, r, sigma, t):
    expected_S = S0 * exp(r*t)
    variance_S = S0 * S0 * (exp(sigma*sigma*t)-1) * exp((2*r+sigma*sigma)*t)
    return expected_S, variance_S


class TestStockSimulations(unittest.TestCase):
    def test_BlackScholesMertonStocks(self):
        bsm_simulator = BlackScholesMertonStockPrices(100, 0.04, 0.02)
        sarray = bsm_simulator.generate_time_series(100, 0.1, 10000)
        average_s = np.mean(sarray[:, -1])

        expected_s, variance_s = expected_stock(100, 0.04, 0.01, 100)

        self.assertAlmostEqual(average_s, expected_s, delta=1.96*sqrt(variance_s))


if __name__ == '__main__':
    unittest.main()
