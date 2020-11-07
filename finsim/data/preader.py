
from datetime import datetime
import os
import logging

import pandas as pd
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
import tables as tb


# native yahoo reader
def extract_online_yahoofinance_data(symbol, startdate, enddate):
    try:
        df = data.DataReader(
            name=symbol,
            data_source='yahoo',
            start=datetime.strptime(startdate, '%Y-%m-%d'),
            end=datetime.strptime(enddate, '%Y-%m-%d')
        )
    except (KeyError, RemoteDataError) as e:
        logging.warning('Symbol {} does not exist between {} and {}.'.format(symbol, startdate, enddate))
        return pd.DataFrame({
            'TimeStamp': [],
            'High': [],
            'Low': [],
            'Open': [],
            'Close': [],
            'Volume': [],
            'Adj Close': []
        })

    oricols = df.columns
    df['TimeStamp'] = df.index
    # df['Date'] = df['TimeStamp'].apply(lambda ts: ts.date())
    df = df[['TimeStamp'] + list(oricols)]

    return df


# yahoo reader with local cache
METATABLE_FILENAME = 'SYMBOL_CACHE.h5'
METATABLE_ROWDES = {
    'symbol': tb.StringCol(shape=(), itemsize=10, dflt='', pos=0),
    'query_startdate': tb.StringCol(shape=(), itemsize=10, dflt='0000-00-00', pos=1),
    'query_enddate': tb.StringCol(shape=(), itemsize=10, dflt='0000-00-00', pos=2),
    'data_startdate': tb.StringCol(shape=(), itemsize=10, dflt='0000-00-00', pos=3),
    'data_enddate': tb.StringCol(shape=(), itemsize=10, dflt='0000-00-00', pos=4)
}


def get_yahoofinance_data(symbol, startdate, enddate, cacheddir=None):
    if cacheddir is None:
        return extract_online_yahoofinance_data(symbol, startdate, enddate)

    if isinstance(cacheddir, str):
        if not os.path.exists(cacheddir) or (os.path.exists(cacheddir) and not os.path.isdir(cacheddir)):
            logging.info('Creating directory: {}'.format(cacheddir))
            os.makedirs(cacheddir)
        if not os.path.exists(os.path.join(cacheddir, METATABLE_FILENAME)):
            logging.info('Creating file: {}'.format(os.path.join(cacheddir, METATABLE_FILENAME)))
            metatable_h5file = tb.open_file(os.path.join(cacheddir, METATABLE_FILENAME), 'w')
            table = metatable_h5file.create_table('/', 'metatable', METATABLE_ROWDES, title='metatable')
        else:
            metatable_h5file = tb.open_file(os.path.join(cacheddir, METATABLE_FILENAME), 'r+')
            table = metatable_h5file.root.metatable

        preexist = False
        for row in table.where('symbol=="{}"'.format(symbol)):
            preexist = True
            if row['query_startdate'].decode('utf-8') <= startdate and row['query_enddate'].decode('utf-8') >= enddate:
                df = pd.read_hdf(os.path.join(cacheddir, '{}.h5'.format(symbol)), 'yahoodata')
                if len(df) > 0:
                    df = df[(df['TimeStamp'] >= startdate) & (df['TimeStamp'] <= enddate)]
                return df

        df = extract_online_yahoofinance_data(symbol, startdate, enddate)
        logging.debug('Caching data for {} from {} to {}'.format(symbol, startdate, enddate))
        df.to_hdf(os.path.join(cacheddir, '{}.h5'.format(symbol)), 'yahoodata')

        if preexist:
            logging.debug('Updating symbol {} in metatable.'.format(symbol))
            for row in table.where('symbol=="{}"'.format(symbol)):
                row['query_startdate'] = startdate
                row['query_enddate'] = enddate
                if len(df) > 0:
                    row['data_startdate'] = datetime.strftime(df['TimeStamp'][0].date(), '%Y-%m-%d')
                    row['data_enddate'] = datetime.strftime(df['TimeStamp'][-1].date(), '%Y-%m-%d')
                else:
                    row['data_startdate'] = '0000-00-00'
                    row['data_enddate'] = ' 0000-00-00'
                row.update()
        else:
            logging.debug('Creating symbol {} in metatable'.format(symbol))
            newrow = table.row
            newrow['symbol'] = symbol
            newrow['query_startdate'] = startdate
            newrow['query_enddate'] = enddate
            if len(df) > 0:
                newrow['data_startdate'] = datetime.strftime(df['TimeStamp'][0].date(), '%Y-%m-%d')
                newrow['data_enddate'] = datetime.strftime(df['TimeStamp'][-1].date(), '%Y-%m-%d')
            else:
                newrow['data_startdate'] = '0000-00-00'
                newrow['data_enddate'] = '0000-00-00'
            newrow.append()

        table.flush()
        metatable_h5file.close()

        return df
    else:
        raise TypeError('Type of cacheddir has to be str, but got {} instead!'.format(type(cacheddir)))
