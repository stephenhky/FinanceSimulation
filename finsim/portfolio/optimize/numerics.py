
import logging
from functools import partial
from itertools import product
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from tqdm import tqdm

from .metrics import sharpe_ratio, mpt_costfunction, mpt_entropy_costfunction
from ...data import get_yahoofinance_data
from ...data.preader import get_dividends_df
from ...estimate.fit import fit_multivariate_BlackScholesMerton_model, fit_BlackScholesMerton_model


def getarrayelementminusminvalue(array, minvalue, index):
    return array[index] - minvalue


def checksumarray(array, total):
    return total - np.sum(array)


def optimized_portfolio_on_sharperatio(r, cov, rf, minweight=0.):
    func = partial(sharpe_ratio, r=r, cov=cov, rf=rf)
    nbstocks = len(r)
    initialguess = np.repeat(1 / nbstocks, nbstocks)
    constraints = [
        {'type': 'ineq', 'fun': partial(getarrayelementminusminvalue, minvalue=minweight, index=i)}
        for i in range(nbstocks)
    ] + [
        {'type': 'eq', 'fun': partial(checksumarray, total=1.)}
    ]
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )


def optimized_portfolio_mpt_costfunction(r, cov, rf, lamb, V0=10.):
    func = partial(mpt_costfunction, r=r, cov=cov, rf=rf, lamb=lamb, V0=V0)
    nbstocks = len(r)
    constraints = [
        {'type': 'ineq', 'fun': partial(getarrayelementminusminvalue, minvalue=0., index=i)}
        for i in range(nbstocks+1)
    ] + [
        {'type': 'ineq', 'fun': partial(checksumarray, total=V0)}
    ]
    initialguess = np.repeat(V0 / (nbstocks+1), nbstocks+1)
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )


def optimized_portfolio_mpt_entropy_costfunction(r, cov, rf, lamb0, lamb1, V=10.):
    func = partial(mpt_entropy_costfunction, r=r, cov=cov, rf=rf, lamb0=lamb0, lamb1=lamb1, V=V)
    nbstocks = len(r)
    constraints = [
        {'type': 'ineq', 'fun': partial(getarrayelementminusminvalue, minvalue=0., index=i)}
        for i in range(nbstocks+1)
    ] + [
        {'type': 'ineq', 'fun': partial(checksumarray, total=V)}
    ]
    initialguess = np.repeat(V / (nbstocks + 1), nbstocks + 1)
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )


def get_BlackScholesMerton_stocks_estimation(
        symbols,
        startdate,
        enddate,
        lazy=False,
        epsilon=1e-10,
        progressbar=True,
        cacheddir=None,
        include_dividends=False
):
    logging.info('Reading financial data...')
    symreadingprogress = tqdm(symbols) if progressbar else symbols
    stocks_data_dfs = [
        get_yahoofinance_data(sym, startdate, enddate, cacheddir=cacheddir)
        for sym in symreadingprogress
    ]

    if include_dividends:
        for i, sym in enumerate(symbols):
            stock_df = stocks_data_dfs[i]
            dividends_df = get_dividends_df(sym)
            dividends_df = dividends_df.rename(columns={'date': 'TimeStamp'})
            dividends_df['Cash'] = np.cumsum(dividends_df['Dividends'].ravel())
            stock_df['TimeStamp'] = stock_df['TimeStamp'].map(lambda ts: datetime.strftime(ts, '%Y-%m-%d'))
            stock_df = stock_df.merge(dividends_df, how='left').ffill().fillna(0)
            stock_df['EffVal'] = stock_df['Close'] + stock_df['Cash']
            stocks_data_dfs[i] = stock_df
    else:
        for i in range(len(symbols)):
            stock_df = stocks_data_dfs[i]
            stock_df['EffVal'] = stock_df['Close'] * 1.
            stocks_data_dfs[i] = stock_df

    # unify the timestamps columns
    logging.info('Unifying timestamps....')
    timestampset = set()
    for stock_df in stocks_data_dfs:
        timestampset = timestampset.union(stock_df['TimeStamp'])
    alltimestamps = sorted(list(timestampset))
    timedf = pd.DataFrame({'AllTime': alltimestamps})

    # wrangle the stock dataframes
    for i, stock_df in enumerate(stocks_data_dfs):
        stock_df = pd.merge(stock_df, timedf, how='right', left_on='TimeStamp', right_on='AllTime').ffill()
        stock_df = stock_df.loc[ ~np.isnan(stock_df['TimeStamp']), stock_df.columns[:-1]]
        stocks_data_dfs[i] = stock_df

    # calculating length
    logging.info('Estimating...')
    max_timearray_ref = 0
    maxlen = max(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)))
    minlen = min(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)) if len(stocks_data_dfs) > 0)   # exclude those stocks that do not exist
    absent_stocks = {sym for sym, df in zip(symbols, stocks_data_dfs) if len(df) == 0}
    logging.debug('maxlen = {}; minlen = {}; absent_stocks: {}'.format(maxlen, minlen, ', '.join(absent_stocks)))

    if maxlen == minlen:
        return fit_multivariate_BlackScholesMerton_model(
            np.array(stocks_data_dfs[max_timearray_ref]['TimeStamp']),
            np.array([
                np.array(stocks_data_dfs[i]['EffVal'])
                for i in range(len(stocks_data_dfs))
            ])
        )

    # TODO: rewrite this whole part
    if maxlen != minlen:
        logging.warning('Not all symbols have data all the way back to {}'.format(startdate))
        max_timearray_ref = [i for i in range(len(stocks_data_dfs)) if maxlen == len(stocks_data_dfs[i])][0]
        logging.warning('Symbols not having whole range of data:')
        for i, symbol in enumerate(symbols):
            if len(stocks_data_dfs[i]) == 0:
                logging.warning('{} has no data between {} and {}'.format(symbol, startdate, enddate))
            elif len(stocks_data_dfs[i]) != maxlen:
                logging.warning('{}: starting from {}'.format(symbol, stocks_data_dfs[i]['TimeStamp'][0].date().strftime('%Y-%m-%d')))
        if lazy:
            logging.warning('Estimation starting from {}'.format(
                stocks_data_dfs[max_timearray_ref]['TimeStamp'][-minlen].date().strftime('%Y-%m-%d')))
            multiprices = np.array([
                np.array(stocks_data_dfs[i]['EffVal'][-minlen:])
                for i in range(len(stocks_data_dfs))
            ])
            return fit_multivariate_BlackScholesMerton_model(
                np.array(stocks_data_dfs[max_timearray_ref]['TimeStamp'][-minlen:]),
                multiprices
            )
        else:
            logging.warning('Estimating with various time lengths...')
            rarray = np.zeros(len(symbols))
            covmat = np.zeros((len(symbols), len(symbols)))
            for i in range(len(symbols)):
                if symbols[i] in absent_stocks:
                    rarray[i] = 0.
                    covmat[i, i] = epsilon   # infinitesimal value
                    continue
                df = stocks_data_dfs[i]
                r, sigma = fit_BlackScholesMerton_model(
                    np.array(df['TimeStamp']),
                    np.array(df['EffVal'])
                )
                rarray[i] = r
                covmat[i, i] = sigma*sigma
            for i, j in product(range(len(symbols)), range(len(symbols))):
                if symbols[i] in absent_stocks or symbols[j] in absent_stocks:
                    covmat[i, j] = 0.
                    covmat[j, i] = 0.
                    continue
                df_i = stocks_data_dfs[i]
                df_j = stocks_data_dfs[j]
                minlen = min(len(df_i), len(df_j))
                try:
                    assert df_i['TimeStamp'][-minlen] == df_j['TimeStamp'][-minlen]
                except AssertionError as e:
                    logging.warning('{}: {}'.format(symbols[i], df_i['TimeStamp'][-minlen]))
                    logging.warning('{}: {}'.format(symbols[j], df_j['TimeStamp'][-minlen]))
                    raise e
                try:
                    assert df_i['TimeStamp'][-1] == df_j['TimeStamp'][-1]
                except AssertionError as e:
                    logging.warning('{}: {}'.format(symbols[i], df_i['TimeStamp'][-1]))
                    logging.warning('{}: {}'.format(symbols[j], df_j['TimeStamp'][-1]))
                    raise e

                ts = df_i['TimeStamp'][-minlen:]
                multiprices = np.array([np.array(df_i['EffVal'][-minlen:]), np.array(df_j['EffVal'][-minlen:])])

                r, cov = fit_multivariate_BlackScholesMerton_model(ts, multiprices)
                covmat[i, j] = cov[0, 1]
                covmat[j, i] = cov[1, 0]
            return rarray, covmat