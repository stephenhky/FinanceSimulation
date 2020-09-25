
from datetime import datetime
import sys

from pandas_datareader import data


def get_yahoofinance_data(symbol, startdate, enddate):
    try:
        df = data.DataReader(
            name=symbol,
            data_source='yahoo',
            start=datetime.strptime(startdate, '%Y-%m-%d'),
            end=datetime.strptime(enddate, '%Y-%m-%d')
        )
    except KeyError as e:
        print('Symbol {} does not exist between {} and {}.'.format(symbol, startdate, enddate),
              file=sys.stderr)
        raise e
    oricols = df.columns
    df['TimeStamp'] = df.index
    # df['Date'] = df['TimeStamp'].apply(lambda ts: ts.date())
    df = df[['TimeStamp'] + list(oricols)]

    return df
