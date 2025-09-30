
import logging
from functools import partial
from itertools import product
from datetime import datetime
from typing import Optional, Literal, Annotated
from os import PathLike

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.optimize import minimize, OptimizeResult
from tqdm import tqdm
import numba as nb

from .metrics import sharpe_ratio, mpt_costfunction, mpt_entropy_costfunction
from ..helper import align_timestamps_stock_dataframes
from ...data import get_yahoofinance_data
from ...data.preader import get_dividends_df
from ...estimate.fit import fit_multivariate_BlackScholesMerton_model, fit_BlackScholesMerton_model, \
    fit_timeweighted_BlackScholesMerton_model, fit_timeweighted_multivariate_BlackScholesMerton_model


@nb.njit(nb.float64(nb.float64[:], nb.float64, nb.int32))
def getarrayelementminusminvalue(
        array: Annotated[NDArray[np.float64], Literal["1D array"]],
        minvalue: float,
        index: int
) -> float:
    """Get the difference between an array element and a minimum value.
    
    Args:
        array: Input array
        minvalue: Minimum value to subtract
        index: Index of the element in the array
        
    Returns:
        float: Difference between array element and minimum value
    """
    return array[index] - minvalue


@nb.njit(nb.float64(nb.float64[:], nb.float64))
def checksumarray(
        array: Annotated[NDArray[np.float64], Literal["1D array"]],
        total: float
) -> float:
    """Calculate the difference between a total and the sum of an array.
    
    Args:
        array: Input array
        total: Total value to compare against
        
    Returns:
        float: Difference between total and sum of array
    """
    return total - np.sum(array)


def optimized_portfolio_on_sharperatio(
        r: Annotated[NDArray[np.float64], Literal["1D array"]],
        cov: Annotated[NDArray[np.float64], Literal["2D array"]],
        rf: float,
        minweight: float=0.
) -> OptimizeResult:
    """Optimize a portfolio based on the Sharpe ratio.
    
    Args:
        r: Array of expected returns
        cov: Covariance matrix
        rf: Risk-free rate
        minweight: Minimum weight for each asset (default: 0.0)
        
    Returns:
        OptimizeResult: Optimization result object
    """
    func = partial(sharpe_ratio, r=r, cov=cov, rf=rf)
    nbstocks = len(r)
    initialguess = np.repeat(1 / nbstocks, nbstocks)
    constraints = [
        {'type': 'ineq', 'fun': partial(getarrayelementminusminvalue, minvalue=minweight, index=i)}
        for i in range(nbstocks)
    ] + [
        {'type': 'eq', 'fun': partial(checksumarray, total=1.)}
    ] + [
        {'type': 'ineq', 'fun': lambda weights: weights[i]}
        for i in range(len(r))
    ]
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )


def optimized_portfolio_mpt_costfunction(
        r: Annotated[NDArray[np.float64], Literal["1D array"]],
        cov: Annotated[NDArray[np.float64], Literal["2D array"]],
        rf: float,
        lamb: float,
        V0: float=10.
) -> OptimizeResult:
    """Optimize a portfolio based on the MPT cost function.
    
    Args:
        r: Array of expected returns
        cov: Covariance matrix
        rf: Risk-free rate
        lamb: Lambda parameter for the cost function
        V0: Initial portfolio value (default: 10.0)
        
    Returns:
        OptimizeResult: Optimization result object
    """
    func = partial(mpt_costfunction, r=r, cov=cov, rf=rf, lamb=lamb, V0=V0)
    nbstocks = len(r)
    constraints = [
        {'type': 'ineq', 'fun': partial(getarrayelementminusminvalue, minvalue=0., index=i)}
        for i in range(nbstocks+1)
    ] + [
        {'type': 'ineq', 'fun': partial(checksumarray, total=V0)}
    ] + [
        {'type': 'ineq', 'fun': lambda weights: weights[i]}
        for i in range(len(r))
    ]
    initialguess = np.repeat(V0 / (nbstocks+1), nbstocks+1)
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )


def optimized_portfolio_mpt_entropy_costfunction(
        r: Annotated[NDArray[np.float64], Literal["1D array"]],
        cov: Annotated[NDArray[np.float64], Literal["2D array"]],
        rf: float,
        lamb0: float,
        lamb1: float,
        V: float=10.
) -> OptimizeResult:
    """Optimize a portfolio based on the MPT entropy cost function.
    
    Args:
        r: Array of expected returns
        cov: Covariance matrix
        rf: Risk-free rate
        lamb0: Lambda 0 parameter for the entropy cost function
        lamb1: Lambda 1 parameter for the entropy cost function
        V: Portfolio value parameter (default: 10.0)
        
    Returns:
        OptimizeResult: Optimization result object
    """
    func = partial(mpt_entropy_costfunction, r=r, cov=cov, rf=rf, lamb0=lamb0, lamb1=lamb1, V=V)
    nbstocks = len(r)
    constraints = [
        {'type': 'ineq', 'fun': partial(getarrayelementminusminvalue, minvalue=0., index=i)}
        for i in range(nbstocks+1)
    ] + [
        {'type': 'ineq', 'fun': partial(checksumarray, total=V)}
    ] + [
        {'type': 'ineq', 'fun': lambda weights: weights[i]}
        for i in range(len(r))
    ]
    initialguess = np.repeat(V / (nbstocks + 1), nbstocks + 1)
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )


def intermediate_wrangle_stock_df_without_dividends(stock_df: pd.DataFrame) -> pd.DataFrame:
    """Process stock data without dividends.
    
    Args:
        stock_df: DataFrame containing stock data
        
    Returns:
        pd.DataFrame: Processed stock DataFrame with 'EffVal' column
    """
    stock_df.loc[:, 'EffVal'] = stock_df['Close'] * 1.
    return stock_df


def intermediate_wrangle_stock_df_with_dividends(stock_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Process stock data with dividends.
    
    Args:
        stock_df: DataFrame containing stock data
        symbol: Stock symbol
        
    Returns:
        pd.DataFrame: Processed stock DataFrame with 'EffVal' column including dividends
    """
    dividends_df = get_dividends_df(symbol)
    dividends_df = dividends_df.rename(columns={'date': 'TimeStamp'})
    dividends_df.loc[:, 'Cash'] = np.cumsum(dividends_df['Dividends'].ravel())
    stock_df.loc[:, 'TimeStamp'] = stock_df['TimeStamp'].map(lambda ts: datetime.strftime(ts, '%Y-%m-%d'))
    stock_df = stock_df.merge(dividends_df, how='left').ffill().fillna(0)
    stock_df.loc[:, 'EffVal'] = stock_df['Close'] + stock_df['Cash']
    stock_df.loc[:, 'TimeStamp'] = stock_df['TimeStamp'].map(lambda ts: datetime.strptime(ts, '%Y-%m-%d'))
    return stock_df


def get_BlackScholesMerton_stocks_estimation(
        symbols: list[str],
        startdate: str,
        enddate: str,
        progressbar: bool=True,
        cacheddir: Optional[PathLike | str]=None,
        include_dividends: bool=False
) -> tuple[Annotated[NDArray[np.float64], Literal["1D array"]], Annotated[NDArray[np.float64], Literal["2D array"]]]:
    """Get Black-Scholes-Merton model estimations for a list of stocks.
    
    Args:
        symbols: List of stock symbols
        startdate: Start date in 'YYYY-MM-DD' format
        enddate: End date in 'YYYY-MM-DD' format
        progressbar: Whether to show a progress bar (default: True)
        cacheddir: Directory for cached data (optional)
        include_dividends: Whether to include dividends in the calculation (default: False)
        
    Returns:
        tuple[NDArray[Shape["*"], Float], NDArray[Shape["*, *"], Float]]: Tuple of (rarray, covmat)
    """
    logging.info('Reading financial data...')
    symreadingprogress = tqdm(symbols) if progressbar else symbols
    stocks_data_dfs = [
        get_yahoofinance_data(sym, startdate, enddate, cacheddir=cacheddir)
        for sym in symreadingprogress
    ]

    if include_dividends:
        for i, symbol in enumerate(symbols):
            stocks_data_dfs[i] = intermediate_wrangle_stock_df_with_dividends(stocks_data_dfs[i], symbol)
    else:
        for i in range(len(symbols)):
            stocks_data_dfs[i] = intermediate_wrangle_stock_df_without_dividends(stocks_data_dfs[i])

    # unify the timestamps columns
    logging.info('Unifying timestamps....')
    stocks_data_dfs = align_timestamps_stock_dataframes(stocks_data_dfs)

    # calculating length
    logging.info('Estimating...')
    max_timearray_ref = 0
    maxlen = max(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)))
    minlen = min(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)) if len(stocks_data_dfs) > 0)   # exclude those stocks that do not exist
    absent_stocks = {sym for sym, df in zip(symbols, stocks_data_dfs) if len(df) == 0}
    logging.debug(f'{maxlen=}; {minlen=}; absent_stocks: {", ".join(absent_stocks)}')

    # same length, directly compare
    if maxlen == minlen:
        return fit_multivariate_BlackScholesMerton_model(
            stocks_data_dfs[max_timearray_ref]['TimeStamp'].ravel(),
            np.array([
                stocks_data_dfs[i]['EffVal'].ravel()
                for i in range(len(stocks_data_dfs))
            ])
        )
    else:    # maxlen != minlen:
        rarray = np.zeros(len(symbols))
        covmat = np.zeros((len(symbols), len(symbols)))

        for i, stock_df in enumerate(stocks_data_dfs):
            r, sigma = fit_BlackScholesMerton_model(stock_df['TimeStamp'].to_numpy(), stock_df['Close'].to_numpy())
            rarray[i] = r
            covmat[i, i] = sigma*sigma

        for i, j in product(range(len(symbols)), range(len(symbols))):
            stock_df_i = stocks_data_dfs[i]
            stock_df_j = stocks_data_dfs[j]
            smallerlen = min(len(stock_df_i), len(stock_df_j))
            _, cov = fit_multivariate_BlackScholesMerton_model(
                stock_df_i.loc[(len(stock_df_i)-smallerlen):, 'TimeStamp'].to_numpy(),
                np.array([
                    stock_df_i.loc[(len(stock_df_i)-smallerlen):, 'Close'].to_numpy(),
                    stock_df_j.loc[(len(stock_df_j)-smallerlen):, 'Close'].to_numpy()
                ])
            )
            covmat[i, j] = cov[0, 1]
            covmat[j, i] = cov[1, 0]

        return rarray, covmat


def get_stocks_timeweighted_estimation(
        symbols: list[str],
        timeweightdf: pd.DataFrame,
        progressbar: bool=True,
        cacheddir: Optional[PathLike | str]=None,
        include_dividends: bool=False
) -> tuple[Annotated[NDArray[np.float64], Literal["1D array"]], Annotated[NDArray[np.float64], Literal["2D array"]]]:
    """Get time-weighted estimations for a list of stocks.
    
    Args:
        symbols: List of stock symbols
        timeweightdf: DataFrame containing timestamps and weights
        progressbar: Whether to show a progress bar (default: True)
        cacheddir: Directory for cached data (optional)
        include_dividends: Whether to include dividends in the calculation (default: False)
        
    Returns:
        tuple[NDArray[Shape["*"], Float], NDArray[Shape["*, *"], Float]]: Tuple of (rarray, covmat)
    """
    logging.info('Parsing weights according to date')
    startdate = timeweightdf['TimeStamp'][0]
    if isinstance(startdate, datetime):
        startdate = startdate.strftime('%Y-%m-%d')
    enddate = timeweightdf['TimeStamp'][len(timeweightdf) - 1]
    if isinstance(enddate, datetime):
        enddate = enddate.strftime('%Y-%m-%d')

    logging.info('Reading financial data...')
    symreadingprogress = tqdm(symbols) if progressbar else symbols
    stocks_data_dfs = [
        get_yahoofinance_data(sym, startdate, enddate, cacheddir=cacheddir)
        for sym in symreadingprogress
    ]

    if include_dividends:
        for i, symbol in enumerate(symbols):
            stocks_data_dfs[i] = intermediate_wrangle_stock_df_with_dividends(stocks_data_dfs[i], symbol)
    else:
        for i in range(len(symbols)):
            stocks_data_dfs[i] = intermediate_wrangle_stock_df_without_dividends(stocks_data_dfs[i])

    # unify the timestamps columns
    logging.info('Unifying timestamps....')
    stocks_data_dfs = align_timestamps_stock_dataframes(stocks_data_dfs)

    # calculating length
    logging.info('Estimating...')
    max_timearray_ref = 0
    maxlen = max(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)))
    minlen = min(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)) if
                 len(stocks_data_dfs) > 0)  # exclude those stocks that do not exist
    absent_stocks = {sym for sym, df in zip(symbols, stocks_data_dfs) if len(df) == 0}
    logging.debug(f'{maxlen=}; {minlen=}; absent_stocks: {", ".join(absent_stocks)}')

    # same length, directly compare
    if maxlen == minlen and maxlen == len(timeweightdf):
        return fit_timeweighted_multivariate_BlackScholesMerton_model(
            np.array(stocks_data_dfs[max_timearray_ref]['TimeStamp']),
            np.array([
                np.array(stocks_data_dfs[i]['EffVal'])
                for i in range(len(stocks_data_dfs))
            ]),
            timeweightdf['weight'].ravel()
        )
    else:
        rarray = np.zeros(len(symbols))
        covmat = np.zeros((len(symbols), len(symbols)))

        for i, stock_df in enumerate(stocks_data_dfs):
            w_stock_df = stock_df.merge(timeweightdf, on='TimeStamp', how='left')
            r, sigma = fit_timeweighted_BlackScholesMerton_model(
                w_stock_df['TimeStamp'].ravel(),
                w_stock_df['Close'].ravel(),
                w_stock_df['weight'].ravel()
            )
            rarray[i] = r
            covmat[i, i] = sigma * sigma

        for i, j in product(range(len(symbols)), range(len(symbols))):
            w_stock_df_i = stocks_data_dfs[i].merge(timeweightdf, on='TimeStamp', how='left')
            w_stock_df_j = stocks_data_dfs[j].merge(timeweightdf, on='TimeStamp', how='left')
            smallerlen = min(len(w_stock_df_i), len(w_stock_df_j))
            _, cov = fit_timeweighted_multivariate_BlackScholesMerton_model(
                w_stock_df_i.loc[(len(w_stock_df_i) - smallerlen):, 'TimeStamp'].ravel(),
                np.array([
                    w_stock_df_i.loc[(len(w_stock_df_i) - smallerlen):, 'Close'].ravel(),
                    w_stock_df_j.loc[(len(w_stock_df_j) - smallerlen):, 'Close'].ravel()
                ]),
                w_stock_df_i.loc[(len(w_stock_df_i) - smallerlen):, 'weight'].ravel()
            )
            covmat[i, j] = cov[0, 1]
            covmat[j, i] = cov[1, 0]

        return rarray, covmat
