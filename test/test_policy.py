
from itertools import product

import unittest
import numpy as np

from finsim.portfolio.optimize.policy import mat_to_list


class TestPolicy(unittest.TestCase):
    def test_mat_to_list(self):
        arr = np.array([[1., 3.], [2., 5.]])
        lists = mat_to_list(arr)
        assert len(lists) == 2
        assert len(lists[0]) == 2
        assert len(lists[1]) == 2
        for i, j in product(range(2), range(2)):
            assert arr[i, j] == lists[i][j]


if __name__ == '__main__':
    unittest.main()