
from datetime import datetime
import os
import logging

import pandas as pd
import yfinance as yf
import tables as tb
from tqdm import tqdm


def extract_online_yahoofinance_data(symbol, startdate, enddate):
    try:
        df = yf.download(
            symbol,
            start=datetime.strptime(startdate, '%Y-%m-%d'),
            end=datetime.strptime(enddate, '%Y-%m-%d')
        )
    except:
        logging.warning('Symbol {} does not exist between {} and {}.'.format(symbol, startdate, enddate))
        return pd.DataFrame({
            'TimeStamp': [],
            'High': [],
            'Low': [],
            'Open': [],
            'Close': [],
            'Adj Close': [],
            'Volume': [],
        })

    oricols = df.columns
    df['TimeStamp'] = df.index
    # df['Date'] = df['TimeStamp'].apply(lambda ts: ts.date())
    df = df[['TimeStamp'] + list(oricols)]

    return df


def extract_batch_online_yahoofinance_data(symbols, startdate, enddate):
    combined_df = yf.download(
        ' '.join(symbols),
        start=datetime.strptime(startdate, '%Y-%m-%d'),
        end=datetime.strptime(enddate, '%Y-%m-%d'),
        group_by='ticker'
    )

    dataframes = {}
    for symbol in symbols:
        try:
            df = combined_df[symbol].copy()
            oricols = df.columns
            df['TimeStamp'] = df.index
            df = df[['TimeStamp'] + list(oricols)]
            dataframes[symbol] = df
        except:
            dataframes[symbol] = pd.DataFrame({
                'TimeStamp': [],
                'High': [],
                'Low': [],
                'Open': [],
                'Close': [],
                'Adj Close': [],
                'Volume': [],
            })

    return dataframes


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
                metatable_h5file.close()
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


def generating_cached_yahoofinance_data(symbols, startdate, enddate, cacheddir, slicebatch=50):
    if not os.path.exists(cacheddir) or (os.path.exists(cacheddir) and not os.path.isdir(cacheddir)):
        logging.info('Creating directory: {}'.format(cacheddir))
        os.makedirs(cacheddir)
    # start as a new file
    logging.info('Creating file: {}'.format(os.path.join(cacheddir, METATABLE_FILENAME)))
    metatable_h5file = tb.open_file(os.path.join(cacheddir, METATABLE_FILENAME), 'w')
    table = metatable_h5file.create_table('/', 'metatable', METATABLE_ROWDES, title='metatable')

    nbsymbols = len(symbols)
    for startidx in tqdm(range(0, nbsymbols, slicebatch)):
        dataframes = extract_batch_online_yahoofinance_data(
            symbols[startidx:min(startidx+slicebatch, nbsymbols)],
            startdate,
            enddate
        )

        for symbol in dataframes:
            df = dataframes[symbol]
            logging.debug('Caching data for {} from {} to {}'.format(symbol, startdate, enddate))
            df.to_hdf(os.path.join(cacheddir, '{}.h5'.format(symbol)), 'yahoodata')

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
