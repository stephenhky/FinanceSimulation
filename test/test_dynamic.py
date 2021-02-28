
import unittest
import logging

from finsim.portfolio import DynamicPortfolio, InsufficientSharesException

logging.basicConfig(level=logging.DEBUG)

class TestDynamicPortfolio(unittest.TestCase):
    def setUp(self):
        self.dynport = DynamicPortfolio({'NVDA': 1}, '2020-01-01')
        self.dynport.trade('2020-02-01', buy_stocks={'NVDA': 2, 'VOO': 1})
        self.dynport.trade('2020-03-01', buy_stocks={'LMT': 1}, sell_stocks={'NVDA': 3})
        self.dynport.trade('2020-04-01', buy_stocks={'COF': 2}, sell_stocks={'LMT': 2})

    def test_orders(self):
        self.assertTrue(self.dynport.is_sorted())
        self.assertEqual(self.dynport.find_cursor_for_date('2018-12-15'), -1)
        self.assertEqual(self.dynport.find_cursor_for_date('2020-01-06'), 0)
        self.assertEqual(self.dynport.find_cursor_for_date('2020-02-14'), 1)
        self.assertEqual(self.dynport.find_cursor_for_date('2020-03-22'), 2)
        self.assertEqual(self.dynport.find_cursor_for_date('2020-04-28'), 3)

    def test_timevalue(self):
        self.assertTrue(self.dynport.get_portfolio_value('2020-03-02'),
                        self.dynport.timeseries[2]['portfolio'].get_portfolio_value('2020-03-02'))

    def test_negative_shares(self):
        self.assertLess(self.dynport.timeseries[3]['portfolio'].symbols_nbshares['LMT'], 0)

    def test_trading_notcheck(self):
        port1 = DynamicPortfolio({'NVDA': 2, 'MSFT': 3}, '2020-01-01')
        port1.trade('2020-01-15', buy_stocks={'LMT': 3, 'NVDA': 1}, sell_stocks={'MSFT': 10})
        self.assertEqual(port1.symbols_nbshares['NVDA'], 3)
        self.assertEqual(port1.symbols_nbshares['LMT'], 3)
        self.assertEqual(port1.symbols_nbshares['MSFT'], -7)

    def test_trading_check_available_fund(self):
        port2 = DynamicPortfolio({'NVDA': 2, 'MSFT': 3}, '2020-01-01')
        port2.trade('2020-01-15', buy_stocks={'LMT': 3, 'NVDA': 1}, sell_stocks={'MSFT': 10},
                    check_valid=True)
        self.assertEqual(port2.symbols_nbshares['NVDA'], 3)
        self.assertEqual(port2.symbols_nbshares['LMT'], 3)
        self.assertEqual(port2.symbols_nbshares['MSFT'], 0)

    def test_trading_raise_exception(self):
        port3 = DynamicPortfolio({'NVDA': 2, 'MSFT': 3}, '2020-01-01')
        with self.assertRaises(InsufficientSharesException):
            port3.trade('2020-01-15', buy_stocks={'LMT': 3, 'NVDA': 1}, sell_stocks={'MSFT': 10},
                        check_valid=True, raise_insufficient_stock_error=True)


if __name__ == '__main__':
    unittest.main()
