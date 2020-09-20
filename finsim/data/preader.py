
from datetime import datetime

from pandas_datareader import data


def get_yahoofinance_data(symbol, startdate, enddate):
    df = data.DataReader(
        name=symbol,
        data_source='yahoo',
        start=datetime.strptime(startdate, '%Y-%m-%d'),
        end=datetime.strptime(enddate, '%Y-%m-%d')
    )
    oricols = df.columns
    df['TimeStamp'] = df.index
    # df['Date'] = df['TimeStamp'].apply(lambda ts: ts.date())
    df = df[['TimeStamp', 'Date'] + list(oricols)]

    return df
