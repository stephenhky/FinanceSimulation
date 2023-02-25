
import unittest

import numpy as np
import pandas as pd

from finsim.tech.ma import get_movingaverage_price_data
from finsim.data.preader import get_yahoofinance_data


class TestMovingAverage(unittest.TestCase):
    def test_sp500(self):
        testdf = get_movingaverage_price_data('VOO', '2016-01-01', '2023-02-01', 200)
        assert testdf.iloc[0]['TimeStamp'].date().strftime('%Y-%m-%d') == '2016-01-04'

    def test_abnb(self):
        testdf = get_movingaverage_price_data('ABNB', '2016-01-01', '2023-02-01', 200)
        assert testdf.iloc[0]['TimeStamp'].date().strftime('%Y-%m-%d') == '2021-06-29'

    def test_average(self):
        testmadf = get_movingaverage_price_data('^GSPC', '2016-01-01', '2023-02-01', 200)
        testdf = get_yahoofinance_data('^GSPC', '2016-01-01', '2023-02-01')

        # get last row
        lastrow_stockdata = testdf.iloc[len(testdf)-1].to_dict()
        lastday_considered = lastrow_stockdata['TimeStamp']
        firstday_considered = lastday_considered - pd.Timedelta('200 days')
        ma = np.mean(testdf.loc[
                         (testdf['TimeStamp'] > firstday_considered) & (testdf['TimeStamp'] <= lastday_considered)
                     ]['Close'])

        # comparison
        malastrow_data = testmadf.iloc[len(testmadf)-1].to_dict()
        assert malastrow_data['TimeStamp'] == testdf.iloc[len(testdf)-1, :]['TimeStamp']
        assert malastrow_data['MA'] == ma


if __name__ == '__main__':
    unittest.main()
