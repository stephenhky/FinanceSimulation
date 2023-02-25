
from datetime import timedelta, datetime

import numpy as np

from ..data.preader import get_yahoofinance_data


def get_movingaverage_price_data(symbol, startdate, enddate, dayswindow, cacheddir=None):
    # making the days difference calculation
    mastartdate = (datetime.strptime(startdate, '%Y-%m-%d') - timedelta(days=dayswindow)).strftime('%Y-%m-%d')
    df = get_yahoofinance_data(symbol, mastartdate, enddate, cacheddir=cacheddir)
    nbrecords = len(df)
    lastday = df.iloc[nbrecords-1, :]['TimeStamp']
    daysdiff = lastday - df['TimeStamp']
    madf = df.copy()
    madf['DaysDiff'] = daysdiff.apply(lambda dd: dd.days)
    maxdaysdiff = madf.iloc[0]['DaysDiff']
    madf['computema'] = maxdaysdiff - madf['DaysDiff'] > dayswindow

    # calculating moving average
    madf['MA'] = madf.apply(
        lambda row: np.mean(
            madf.loc[(madf['DaysDiff'] >= row['DaysDiff']) & (madf['DaysDiff'] < row['DaysDiff']+dayswindow), 'Close']
        ),
        axis=1
    )

    return madf.loc[madf['computema'], ['TimeStamp', 'MA']]
