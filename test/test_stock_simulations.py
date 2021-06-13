
import unittest
from math import exp, sqrt

import numpy as np

from finsim.simulation.stock import BlackScholesMertonStockPrices, HestonStockPrices, MertonJumpDiffusionStockPrices
from finsim.estimate.fit import fortranfit


def expected_bsm_stock(S0, r, sigma, t):
    expected_S = S0 * exp(r*t)
    variance_S = S0 * S0 * (exp(sigma*sigma*t)-1) * exp((2*r+sigma*sigma)*t)
    return expected_S, variance_S


class TestStockSimulations(unittest.TestCase):
    def backend_test_BlackScholesMertonStocks(self, S, r, sigma, T, dt, nbsimulations):
        ts = np.linspace(0, T, int(T // dt) + 1)

        bsm_simulator = BlackScholesMertonStockPrices(S, r, sigma)
        sarray = bsm_simulator.generate_time_series(T, dt, nbsimulations=nbsimulations)
        rsigma_array = np.array([fortranfit.f90_fit_blackscholesmerton_model(ts, sarray[i, :])
                                 for i in range(nbsimulations)])
        average_r = np.nanmean(rsigma_array[:, 0])
        std_r = np.nanstd(rsigma_array[:, 0])
        average_sigma = np.nanmean(rsigma_array[:, 1])
        std_sigma = np.nanstd(rsigma_array[:, 1])

        self.assertAlmostEqual(float(average_r), r, delta=2.576*std_r)
        self.assertAlmostEqual(float(average_sigma), sigma, delta=2.576*std_sigma)

    def test_BlackScholesMertonStocks(self):
        self.backend_test_BlackScholesMertonStocks(100., 0.04, 0.02, 10, 0.1, 100000)

        for S in np.random.normal(100., 5., size=3):
            for r in np.random.normal(0.1, 0.01, size=3):
                for sigma in np.random.uniform(0.001, 0.05, size=3):
                    self.backend_test_BlackScholesMertonStocks(S, r, sigma, 10, 0.1, 1000)

    def test_HestonStocks(self):
        h_simulator = HestonStockPrices(100, 0.01, 0.01, 0.01, 0.1, 0.0001, 0.2)
        h_stocks = h_simulator.generate_time_series(100, 0.1, nbsimulations=100000)[0]

        self.assertAlmostEqual(
            float(np.nanmean(h_stocks[:, 100])),
            110.52,
            delta=2.576*0.11
        )
        self.assertAlmostEqual(
            float(np.nanmean(h_stocks[:, 250])),
            128.41,
            delta=2.576*0.23
        )
        self.assertAlmostEqual(
            float(np.nanmean(h_stocks[:, 500])),
            164.92,
            delta=2.576*0.38
        )
        self.assertAlmostEqual(
            float(np.nanmean(h_stocks[:, 750])),
            211.80,
            delta=2.576*0.56
        )
        self.assertAlmostEqual(
            float(np.nanmean(h_stocks[:, 999])),
            271.72,
            delta=2.576*0.80
        )

    def test_MertonJumpDiffusionStocks(self):
        mjd_simulator = MertonJumpDiffusionStockPrices(100, 0.01, 0.01, 0.01, 0.01, 0.001)
        mjd_stocks = [
            np.mean(mjd_simulator.generate_time_series(100, 0.1, nbsimulations=1000), axis=0)
            for _ in range(100)
        ]

        self.assertAlmostEqual(
            float(np.mean([stock[250] for stock in mjd_stocks])),
            197.97,
            delta=1.96*6.37
        )
        self.assertAlmostEqual(
            float(np.mean([stock[500] for stock in mjd_stocks])),
            393.35,
            delta=1.96*28.15
        )
        self.assertAlmostEqual(
            float(np.mean([stock[750] for stock in mjd_stocks])),
            773.20,
            delta=1.96*67.73
        )
        self.assertAlmostEqual(
            float(np.mean([stock[999] for stock in mjd_stocks])),
            1523.06,
            delta=1.96*169.68
        )


if __name__ == '__main__':
    unittest.main()
