
import unittest
from datetime import datetime, timedelta
from math import sqrt

import numpy as np

from finsim.simulation.stock import BlackScholesMertonStockPrices
from finsim.estimate.fit import fit_BlackScholesMerton_model


class TestParameterFitting(unittest.TestCase):
    def fit_differentbackend(self, lowlevellang):
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
            for r, sigma in map(lambda i: fit_BlackScholesMerton_model(timestamps, prices[i, :], unit='day', lowlevellang=lowlevellang), range(1000))
        ]
        mean_r = np.mean([item[0] for item in info])
        mean_sigma = np.mean([item[1] for item in info])

        self.assertAlmostEqual(mean_r, 0.04, delta=1.96*0.02/sqrt(1000))
        self.assertAlmostEqual(mean_sigma, 0.02, delta=1.96*0.02/sqrt(1000))

    def test_fit_Cython(self):
        self.fit_differentbackend('C')

    def test_fit_fortran(self):
        self.fit_differentbackend('F')

    def test_fit_python(self):
        self.fit_differentbackend('P')


if __name__ == '__main__':
    unittest.main()
