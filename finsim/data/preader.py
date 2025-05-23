
from datetime import datetime, timedelta
import os
import logging
from time import sleep
import glob
from functools import lru_cache
import threading
import traceback

import pandas as pd
import tables
import yfinance as yf
import tables as tb
from tqdm import tqdm


def extract_online_yahoofinance_data(symbol, startdate, enddate):
    try:
        df = yf.download(
            symbol,
            start=datetime.strptime(startdate, '%Y-%m-%d'),
            end=datetime.strptime(enddate, '%Y-%m-%d') + timedelta(days=1),
            auto_adjust=False
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
    if isinstance(oricols, pd.core.indexes.multi.MultiIndex):
        df.columns = [col[0] for col in oricols]
        oricols = df.columns
    df['TimeStamp'] = df.index
    df = df.loc[:, ['TimeStamp'] + list(oricols)].reset_index()

    return df


def extract_batch_online_yahoofinance_data(symbols, startdate, enddate, threads=True):
    combined_df = yf.download(
        ' '.join(symbols),
        start=datetime.strptime(startdate, '%Y-%m-%d'),
        end=datetime.strptime(enddate, '%Y-%m-%d') + timedelta(days=1),
        group_by='ticker',
        threads=threads
    )

    dataframes = {}
    for symbol in symbols:
        try:
            df = combined_df[symbol].copy()
            oricols = df.columns
            df['TimeStamp'] = df.index
            df = df.loc[:, ['TimeStamp'] + list(oricols)]
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
                df = pd.read_hdf(os.path.join(cacheddir, '{}.h5'.format(symbol)), key='yahoodata')
                if len(df) > 0:
                    df = df[(df['TimeStamp'] >= startdate) & (df['TimeStamp'] <= enddate)]
                metatable_h5file.close()
                df = df[~df['Close'].isna()]
                return df

        df = extract_online_yahoofinance_data(symbol, startdate, enddate)
        logging.debug('Caching data for {} from {} to {}'.format(symbol, startdate, enddate))
        df.to_hdf(os.path.join(cacheddir, '{}.h5'.format(symbol)), key='yahoodata')

        if preexist:
            logging.debug('Updating symbol {} in metatable.'.format(symbol))
            for row in table.where('symbol=="{}"'.format(symbol)):
                row['query_startdate'] = startdate
                row['query_enddate'] = enddate
                if len(df) > 0:
                    row['data_startdate'] = datetime.strftime(df['TimeStamp'].to_list()[0].date(), '%Y-%m-%d')
                    row['data_enddate'] = datetime.strftime(df['TimeStamp'].to_list()[-1].date(), '%Y-%m-%d')
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
                newrow['data_startdate'] = datetime.strftime(df['TimeStamp'].to_list()[0].date(), '%Y-%m-%d')
                newrow['data_enddate'] = datetime.strftime(df['TimeStamp'].to_list()[-1].date(), '%Y-%m-%d')
            else:
                newrow['data_startdate'] = '0000-00-00'
                newrow['data_enddate'] = '0000-00-00'
            newrow.append()

        table.flush()
        metatable_h5file.close()

        df = df[~df['Close'].isna()]

        return df
    else:
        raise TypeError('Type of cacheddir has to be str, but got {} instead!'.format(type(cacheddir)))


@lru_cache(maxsize=256)
def get_symbol_closing_price(symbol, datestr, epsilon=1e-10, cacheddir=None, backtrack=False):
    df = get_yahoofinance_data(symbol, datestr, datestr, cacheddir=cacheddir)
    if len(df) == 0:
        if backtrack:
            prevdatestr = datetime.strftime(datetime.strptime(datestr, '%Y-%m-%d') - timedelta(days=1), '%Y-%m-%d')
            return get_symbol_closing_price(symbol, prevdatestr, epsilon=epsilon, cacheddir=cacheddir, backtrack=True)
        else:
            raise IndexError('Price not found!')
    else:
        return df['Close'][0]


def finding_missing_symbols_in_cache(symbols, startdate, enddate, cacheddir):
    if not os.path.exists(os.path.join(cacheddir, METATABLE_FILENAME)):
        return symbols

    # in table
    metatable = pd.read_hdf(os.path.join(cacheddir, METATABLE_FILENAME), 'metatable')
    existing_within_range_symbols = list(
        metatable['symbol'][
            (metatable['query_startdate'] <= startdate) & (metatable['query_enddate'] >= enddate)
        ]
    )
    if logging.root.level >= logging.DEBUG:
        logging.debug('exisiting within range symbols')
        for symbol in existing_within_range_symbols:
            logging.debug('\t{}'.format(symbol))

    # check what are in the cached directories
    existing_symbols = [
        os.path.basename(filepath)
        for filepath in glob.glob(os.path.join(cacheddir, '*.h5'))
    ]
    existing_symbols = [filename[:-3] for filename in existing_symbols if filename != METATABLE_FILENAME]
    if logging.root.level >= logging.DEBUG:
        logging.debug('exisiting symbols')
        for symbol in existing_symbols:
            logging.debug('\t{}'.format(symbol))

    existing_valid_symbols = set(existing_within_range_symbols) & set(existing_symbols)

    return sorted(list(set(symbols) - set(existing_valid_symbols)))


def dataframe_to_hdf(df, filepath, key):
    df.to_hdf(filepath, key=key)


def generating_cached_yahoofinance_data(
        symbols,
        startdate,
        enddate,
        cacheddir,
        slicebatch=50,
        waittime=1,
        yfinance_multithreads=False,
        io_multithreads=False
):
    tocache_symbols = finding_missing_symbols_in_cache(symbols, startdate, enddate, cacheddir)

    logging.info('Total number of symbols: {}'.format(len(symbols)))
    logging.info('Total number of symbols needed to cache: {}'.format(len(tocache_symbols)))
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

    nbsymbols = len(tocache_symbols)
    for startidx in tqdm(range(0, nbsymbols, slicebatch)):
        success = False
        while not success:
            try:
                dataframes = extract_batch_online_yahoofinance_data(
                    tocache_symbols[startidx:min(startidx + slicebatch, nbsymbols)],
                    startdate,
                    enddate,
                    threads=yfinance_multithreads
                )
                success = True
            except:
                sleep(waittime)

        writing_threads = []
        for symbol in dataframes:
            df = dataframes[symbol]
            df = df[~df['Close'].isna()]
            if len(df) > 0:
                thissymbol_startdate = datetime.strftime(df['TimeStamp'].to_list()[0].date(), '%Y-%m-%d')
                thissymbol_enddate = datetime.strftime(df['TimeStamp'].to_list()[-1].date(), '%Y-%m-%d')
            else:
                thissymbol_startdate = '0000-00-00'
                thissymbol_enddate = '0000-00-00'

            logging.debug('Caching data for {} from {} to {}'.format(symbol, startdate, enddate))
            if not io_multithreads:
                dataframe_to_hdf(df, os.path.join(cacheddir, '{}.h5'.format(symbol)), key='yahoodata')
            else:
                thread = threading.Thread(
                    target=dataframe_to_hdf,
                    args=(df, os.path.join(cacheddir, '{}.h5'.format(symbol)), 'yahoodata')
                )
                thread.start()
                writing_threads.append(thread)

                try:
                    logging.debug('Creating symbol {} in metatable'.format(symbol))
                    newrow = table.row
                    newrow['symbol'] = symbol
                    newrow['query_startdate'] = startdate
                    newrow['query_enddate'] = enddate
                    newrow['data_startdate'] = thissymbol_startdate
                    newrow['data_enddate'] = thissymbol_enddate
                    newrow.append()

                except tables.HDF5ExtError as e:
                    logging.error('Cannot append record for symbol {}'.format(symbol))
                    traceback.print_exc()
                    continue

            table.flush()

        if io_multithreads:
            for thread in writing_threads:
                thread.join()

    metatable_h5file.close()


@lru_cache(maxsize=30)
def get_dividends_df(symbol):
    ticker = yf.Ticker(symbol)
    df = pd.DataFrame(ticker.dividends)
    df['TimeStamp'] = df.index.map(lambda item: datetime.strftime(item, '%Y-%m-%d'))
    df = df.loc[:, ['TimeStamp', 'Dividends']]
    return df
