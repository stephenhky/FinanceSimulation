
import unittest

from finsim.data.preader import get_symbol_closing_price


class TestDynamicPortfolio(unittest.TestCase):
    def test_get_voo_backtrack(self):
        self.assertAlmostEqual(
            get_symbol_closing_price('VOO', '2016-01-01', backtrack=True),
            186.92999267578125
        )

    def test_get_voo_nobacktrack(self):
        self.assertRaises(
            IndexError,
            get_symbol_closing_price,
            'VOO',
            '2016-01-01',
            backtrack=False
        )
        self.assertAlmostEqual(
            get_symbol_closing_price('VOO', '2015-12-31', backtrack=True),
            186.92999267578125
        )


if __name__ == '__main__':
    unittest.main()
