
import unittest

import numpy as np

from finsim.portfolio.optimize import metrics


class TestMetrics(unittest.TestCase):
    def test_sharpe_ratio(self):
        weights = np.array([0.3, 0.5, 0.2])
        r = np.array([0.1, 0.05, 0.15])
        cov = np.array([[1., 0.4, -0.3], [0.4, 1., 0.05], [-0.3, 0.05, 1.]])
        rf = 0.001
        self.assertAlmostEqual(
            metrics.sharpe_ratio(weights, r, cov, rf),
            0.12200850769256662,
            delta=1e-6
        )

    def test_costfunction(self):
        weights = np.array([0.05, 0.3, 0.5, 0.2])
        r = np.array([0.1, 0.05, 0.15])
        cov = np.array([[1., 0.4, -0.3], [0.4, 1., 0.05], [-0.3, 0.05, 1.]])
        rf = 0.001
        self.assertAlmostEqual(
            metrics.mpt_costfunction(weights, r, cov, rf, 0.1, V0=10),
            0.077475,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_costfunction(weights, r, cov, rf, 0.2, V0=10),
            0.059750000000000004,
            delta=1e-6
        )

    def test_entropycostfunction(self):
        weights = np.array([0.05, 0.3, 0.5, 0.2])
        r = np.array([0.1, 0.05, 0.15])
        cov = np.array([[1., 0.4, -0.3], [0.4, 1., 0.05], [-0.3, 0.05, 1.]])
        rf = 0.001
        self.assertAlmostEqual(
            metrics.mpt_entropy_costfunction(weights, r, cov, rf, 0.1, 0.01, V=10),
            0.0897721895621705,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_entropy_costfunction(weights, r, cov, rf, 0.1, 0.1, V=10),
            0.20044689562170498,
            delta=1e-6
        )


if __name__ == '__main__':
    unittest.main()
