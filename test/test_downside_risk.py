
import unittest

import numpy as np

from finsim.estimate import risk


class TestDownsideRisk(unittest.TestCase):
    def test_nodownward_movement(self):
        ts = np.linspace(0, 10, 11)
        S = np.linspace(1.0, 1.8, 11)
        downside_risk = risk.estimate_downside_risk(ts, S, 0.0)
        self.assertAlmostEqual(downside_risk, 0.0)

    def test_somedownward_movement(self):
        r_fake_array = np.array([0.1, 0.2, -0.1, -0.1, 0.3, 0.4])
        S_array = np.exp(np.cumsum(r_fake_array))
        ts = np.linspace(0, 5, 6)
        downside_risk = risk.estimate_downside_risk(ts, S_array, 0.0)
        self.assertAlmostEqual(downside_risk, np.sqrt(0.02/5))


if __name__ == '__main__':
    unittest.main()
