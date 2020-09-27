import sys
from functools import partial
from itertools import product

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from tqdm import tqdm

from ..data.preader import get_yahoofinance_data
from ..estimate.fit import fit_multivariate_BlackScholesMerton_model, fit_BlackScholesMerton_model
from .metrics import sharpe_ratio


def get_symbol_closing_price(symbol, datestr, epsilon=1e-10):
    try:
        return get_yahoofinance_data(symbol, datestr, datestr)['Close'][0]
    except KeyError:
        return epsilon


def get_BlackScholesMerton_stocks_estimation(symbols, startdate, enddate, lazy=False, epsilon=1e-10):
    print('Reading financial data...')
    stocks_data_dfs = [
        get_yahoofinance_data(sym, startdate, enddate)
        for sym in tqdm(symbols)
    ]

    print('Estimating...')
    max_timearray_ref = 0
    maxlen = max(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)))
    minlen = min(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)) if len(stocks_data_dfs) > 0)   # exclude those stocks that do not exist
    absent_stocks = {sym for sym, df in zip(symbols, stocks_data_dfs) if len(df) == 0}
    if maxlen == minlen:
        return fit_multivariate_BlackScholesMerton_model(
            np.array(stocks_data_dfs[max_timearray_ref]['TimeStamp']),
            np.array([
                np.array(stocks_data_dfs[i]['Close'])
                for i in range(len(stocks_data_dfs))
            ])
        )
    if maxlen != minlen:
        print('Not all symbols have data all the way back to {}'.format(startdate), file=sys.stderr)
        max_timearray_ref = [i for i in range(len(stocks_data_dfs)) if maxlen == len(stocks_data_dfs[i])][0]
        print('Symbols not having whole range of data:', file=sys.stderr)
        for i, symbol in enumerate(symbols):
            if len(stocks_data_dfs[i]) == 0:
                print('{} has no data between {} and {}'.format(symbol, startdate, enddate), file=sys.stderr)
            elif len(stocks_data_dfs[i]) != maxlen:
                print('{}: starting from {}'.format(symbol, stocks_data_dfs[i]['TimeStamp'][0].date().strftime('%Y-%m-%d')),
                      file=sys.stderr)
        if lazy:
            print('Estimation starting from {}'.format(
                stocks_data_dfs[max_timearray_ref]['TimeStamp'][-minlen].date().strftime('%Y-%m-%d')),
                  file=sys.stderr)
            multiprices = np.array([
                np.array(stocks_data_dfs[i]['Close'][-minlen:])
                for i in range(len(stocks_data_dfs))
            ])
            return fit_multivariate_BlackScholesMerton_model(
                np.array(stocks_data_dfs[max_timearray_ref]['TimeStamp'][-minlen:]),
                multiprices
            )
        else:
            print('Estimating with various time lengths...', file=sys.stderr)
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
                    np.array(df['Close'])
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
                assert df_i['TimeStamp'][-minlen] == df_j['TimeStamp'][-minlen]
                assert df_i['TimeStamp'][-1] == df_j['TimeStamp'][-1]

                ts = df_i['TimeStamp'][-minlen:]
                multiprices = np.array([np.array(df_i['Close'][-minlen:]), np.array(df_j['Close'][-minlen:])])

                r, cov = fit_multivariate_BlackScholesMerton_model(ts, multiprices)
                covmat[i, j] = cov[0, 1]
                covmat[j, i] = cov[1, 0]
            return rarray, covmat


def optimized_portfolio_on_sharperatio(r, cov, rf, minweight=0.):
    func = partial(sharpe_ratio, r=r, cov=cov, rf=rf)
    nbstocks = len(r)
    initialguess = np.repeat(1 / nbstocks, nbstocks)
    constraints = [
        LinearConstraint(np.eye(nbstocks), minweight, 1.),
        LinearConstraint(np.array([np.repeat(1, nbstocks)]), 1, 1)
    ]
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )