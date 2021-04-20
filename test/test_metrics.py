
import unittest

import numpy as np

from finsim.portfolio.optimize import metrics


class TestMetrics(unittest.TestCase):
    def test_sharpe_ratio(self):
        weights = np.array([0.3, 0.5, 0.2])
        r = np.array([0.1, 0.05, 0.15])
        cov = np.array([[1., 0.4, -0.3], [0.4, 1., 0.05], [-0.3, 0.05, 1.]])
        rf = 0.001
        expected_sharpe_ratio = 0.12200850769256662

        self.assertAlmostEqual(
            metrics.sharpe_ratio(weights, r, cov, rf, lowlevellang='C'),
            expected_sharpe_ratio,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.sharpe_ratio(weights, r, cov, rf, lowlevellang='P'),
            expected_sharpe_ratio,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.sharpe_ratio(weights, r, cov, rf, lowlevellang='F'),
            expected_sharpe_ratio,
            delta=1e-6
        )

    def test_costfunction(self):
        weights = np.array([0.05, 0.3, 0.5, 0.2])
        r = np.array([0.1, 0.05, 0.15])
        cov = np.array([[1., 0.4, -0.3], [0.4, 1., 0.05], [-0.3, 0.05, 1.]])
        rf = 0.001

        lamb01cost = 0.077475
        self.assertAlmostEqual(
            metrics.mpt_costfunction(weights, r, cov, rf, 0.1, V0=10, lowlevellang='C'),
            lamb01cost,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_costfunction(weights, r, cov, rf, 0.1, V0=10, lowlevellang='P'),
            lamb01cost,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_costfunction(weights, r, cov, rf, 0.1, V0=10, lowlevellang='F'),
            lamb01cost,
            delta=1e-6
        )

        lamb02cost = 0.059750000000000004
        self.assertAlmostEqual(
            metrics.mpt_costfunction(weights, r, cov, rf, 0.2, V0=10, lowlevellang='C'),
            lamb02cost,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_costfunction(weights, r, cov, rf, 0.2, V0=10, lowlevellang='P'),
            lamb02cost,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_costfunction(weights, r, cov, rf, 0.2, V0=10, lowlevellang='F'),
            lamb02cost,
            delta=1e-6
        )

    def test_entropycostfunction(self):
        weights = np.array([0.05, 0.3, 0.5, 0.2])
        r = np.array([0.1, 0.05, 0.15])
        cov = np.array([[1., 0.4, -0.3], [0.4, 1., 0.05], [-0.3, 0.05, 1.]])
        rf = 0.001

        case1cost = 0.08170682914813143
        self.assertAlmostEqual(
            metrics.mpt_entropy_costfunction(weights, r, cov, rf, 0.1, 0.01, V=10, lowlevellang='C'),
            case1cost,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_entropy_costfunction(weights, r, cov, rf, 0.1, 0.01, V=10, lowlevellang='P'),
            case1cost,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_entropy_costfunction(weights, r, cov, rf, 0.1, 0.01, V=10, lowlevellang='F'),
            case1cost,
            delta=1e-6
        )

        case2cost = 0.11979329148131437
        self.assertAlmostEqual(
            metrics.mpt_entropy_costfunction(weights, r, cov, rf, 0.1, 0.1, V=10, lowlevellang='C'),
            case2cost,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_entropy_costfunction(weights, r, cov, rf, 0.1, 0.1, V=10, lowlevellang='P'),
            case2cost,
            delta=1e-6
        )
        self.assertAlmostEqual(
            metrics.mpt_entropy_costfunction(weights, r, cov, rf, 0.1, 0.1, V=10, lowlevellang='F'),
            case2cost,
            delta=1e-6
        )


if __name__ == '__main__':
    unittest.main()
