
from itertools import product

import unittest
import numpy as np
import pandas as pd

from finsim.portfolio.optimize.policy import (
    mat_to_list,
    OptimizedWeightingPolicyUsingMPTSharpeRatio,
    OptimizedWeightingPolicyUsingMPTCostFunction,
    OptimizedWeightingPolicyUsingMPTEntropyCostFunction,
)


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.rf = 0.01
        self.r = np.array([0.12, 0.08, 0.15])
        self.cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.002],
            [0.005, 0.002, 0.05]
        ])
        self.symbols = ['AAPL', 'MSFT', 'GOOG']

    def test_mat_to_list(self):
        arr = np.array([[1., 3.], [2., 5.]])
        lists = mat_to_list(arr)
        assert len(lists) == 2
        assert len(lists[0]) == 2
        assert len(lists[1]) == 2
        for i, j in product(range(2), range(2)):
            assert arr[i, j] == lists[i][j]

    def test_sharpe_ratio_policy(self):
        policy = OptimizedWeightingPolicyUsingMPTSharpeRatio(
            self.rf, self.r, self.cov, symbols=self.symbols, minweight=0.1
        )
        self.assertTrue(policy.optimized)
        self.assertEqual(policy.policytype, 'OptimizedWeightingPolicyUsingMPTSharpeRatio')
        self.assertEqual(policy.portfolio_symbols, self.symbols)
        self.assertAlmostEqual(float(np.sum(policy.weights)), 1.0, places=5)
        self.assertTrue(np.all(policy.weights >= 0.1 - 1e-5))
        self.assertGreater(policy.sharpe_ratio, 0.0)
        self.assertGreater(policy.portfolio_yield, 0.0)
        self.assertGreater(policy.volatility, 0.0)
        self.assertIsInstance(policy.named_correlation_matrix, pd.DataFrame)
        self.assertIn('sharpe_ratio', policy.portfolio_summary)

    def test_mpt_costfunction_policy(self):
        policy = OptimizedWeightingPolicyUsingMPTCostFunction(
            self.rf, self.r, self.cov, symbols=self.symbols, lamb=0.1, V0=10.0
        )
        self.assertTrue(policy.optimized)
        self.assertEqual(policy.policytype, 'OptimizedWeightingPolicyUsingMPTCostFunction')
        self.assertAlmostEqual(float(np.sum(policy.weights)), 1.0, places=5)
        self.assertIn('mpt_costfunction', policy.portfolio_summary)
        self.assertEqual(policy.portfolio_summary['V0'], 10.0)
        self.assertEqual(policy.portfolio_summary['lamb'], 0.1)

    def test_mpt_entropy_costfunction_policy(self):
        policy = OptimizedWeightingPolicyUsingMPTEntropyCostFunction(
            self.rf, self.r, self.cov, symbols=self.symbols, lamb0=0.1, lamb1=0.05, V=10.0
        )
        self.assertTrue(policy.optimized)
        self.assertEqual(policy.policytype, 'OptimizedWeightingPolicyUsingMPTEntropyCostFunction')
        self.assertAlmostEqual(float(np.sum(policy.weights)), 1.0, places=5)
        self.assertIn('mpt_entropy_costfunction', policy.portfolio_summary)
        self.assertEqual(policy.portfolio_summary['V'], 10.0)
        self.assertEqual(policy.portfolio_summary['lamb0'], 0.1)
        self.assertEqual(policy.portfolio_summary['lamb1'], 0.05)


if __name__ == '__main__':
    unittest.main()