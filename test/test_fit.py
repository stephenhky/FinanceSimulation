
import unittest
from datetime import datetime, timedelta
from math import sqrt

import numpy as np

from finsim.simulation.stock import BlackScholesMertonStockPrices
from finsim.estimate.fit import (
    fit_BlackScholesMerton_model,
    fit_multivariate_BlackScholesMerton_model,
    fit_timeweighted_BlackScholesMerton_model,
    fit_timeweighted_multivariate_BlackScholesMerton_model,
)


class TestParameterFitting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_fit(self):
        bsm_simulator = BlackScholesMertonStockPrices(100, 0.04, 0.02)
        prices = bsm_simulator.generate_time_series(100, 1, nbsimulations=1000)
        timestamps = np.array(
            [
                datetime(2020, 1, 1, 0, 0, 0) + i*timedelta(1)
                for i in range(101)
            ],
            dtype='datetime64[s]'
        )
        info = [
            (r, sigma)
            for r, sigma in map(lambda i: fit_BlackScholesMerton_model(timestamps, prices[i, :], unit='day'), range(1000))
        ]
        mean_r = np.mean([item[0] for item in info])
        mean_sigma = np.mean([item[1] for item in info])

        self.assertAlmostEqual(mean_r, 0.04, delta=1.96*0.02/sqrt(1000))
        self.assertAlmostEqual(mean_sigma, 0.02, delta=1.96*0.02/sqrt(1000))

    def test_multivariate_and_timeweighted_fit(self):
        timestamps = np.array(
            [datetime(2020, 1, 1, 0, 0, 0) + i * timedelta(1) for i in range(51)],
            dtype='datetime64[s]'
        )
        sim1 = BlackScholesMertonStockPrices(100.0, 0.03, 0.01)
        sim2 = BlackScholesMertonStockPrices(50.0, 0.05, 0.02)
        p1 = sim1.generate_time_series(50, 1, nbsimulations=1)[0]
        p2 = sim2.generate_time_series(50, 1, nbsimulations=1)[0]
        multiprices = np.vstack([p1, p2])

        r1, sigma1 = fit_BlackScholesMerton_model(timestamps, p1, unit='day')
        r2, sigma2 = fit_BlackScholesMerton_model(timestamps, p2, unit='day')
        r_multi, cov_multi = fit_multivariate_BlackScholesMerton_model(timestamps, multiprices, unit='day')

        np.testing.assert_allclose(r_multi, np.array([r1, r2]), rtol=1e-6)
        self.assertAlmostEqual(float(np.sqrt(cov_multi[0, 0])), sigma1, places=6)
        self.assertAlmostEqual(float(np.sqrt(cov_multi[1, 1])), sigma2, places=6)

        uniform_weights = np.ones(len(timestamps))
        rw1, sigmaw1 = fit_timeweighted_BlackScholesMerton_model(timestamps, p1, uniform_weights, unit='day')
        self.assertAlmostEqual(rw1, r1, places=6)
        self.assertAlmostEqual(sigmaw1, sigma1, places=6)

        rw_multi, covw_multi = fit_timeweighted_multivariate_BlackScholesMerton_model(
            timestamps, multiprices, uniform_weights, unit='day'
        )
        np.testing.assert_allclose(rw_multi, r_multi, rtol=1e-6)
        np.testing.assert_allclose(covw_multi, cov_multi, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
