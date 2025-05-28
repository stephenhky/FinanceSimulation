
from datetime import datetime, timedelta
import warnings
from typing import Union
from os import PathLike

import numpy as np
import pandas as pd

from .optimize.numerics import get_BlackScholesMerton_stocks_estimation, get_stocks_timeweighted_estimation
from .portfolio import OptimizedPortfolio
from .optimize.policy import OptimizedWeightingPolicyUsingMPTSharpeRatio, \
    OptimizedWeightingPolicyUsingMPTCostFunction, OptimizedWeightingPolicyUsingMPTEntropyCostFunction


def get_optimized_portfolio_on_sharpe_ratio(
        rf: float,
        symbols: list[str],
        totalworth: float,
        presetdate: str,
        estimating_startdate: str,
        estimating_enddate: str,
        minweight: float=0.,
        lazy: bool=False,
        cacheddir: Union[PathLike, str]=None,
        include_dividends: bool=False
) -> OptimizedPortfolio:
    if lazy:
        warnings.warn('Setting lazy=True is meaningless! Parameter deprecated!')
    r, cov = get_BlackScholesMerton_stocks_estimation(
        symbols,
        estimating_startdate,
        estimating_enddate,
        cacheddir=cacheddir,
        include_dividends=include_dividends
    )
    optimized_weighting_policy = OptimizedWeightingPolicyUsingMPTSharpeRatio(rf, r, cov, symbols, minweight=minweight)
    optimized_portfolio = OptimizedPortfolio(optimized_weighting_policy, totalworth, presetdate, cacheddir=cacheddir)
    return optimized_portfolio


def get_optimized_portfolio_on_mpt_costfunction(
        rf: float,
        symbols: list[str],
        totalworth: float,
        presetdate: str,
        estimating_startdate: str,
        estimating_enddate: str,
        lamb: float,
        V0: float=10.,
        cacheddir: Union[PathLike, str]=None,
        include_dividends: bool=False
) -> OptimizedPortfolio:
    r, cov = get_BlackScholesMerton_stocks_estimation(
        symbols,
        estimating_startdate,
        estimating_enddate,
        cacheddir=cacheddir,
        include_dividends=include_dividends
    )
    optimized_weighting_policy = OptimizedWeightingPolicyUsingMPTCostFunction(rf, r, cov, symbols, lamb, V0=V0)
    optimized_portfolio = OptimizedPortfolio(optimized_weighting_policy, totalworth, presetdate, cacheddir=cacheddir)
    return optimized_portfolio


def get_optimized_portfolio_on_mpt_entropy_costfunction(
        rf: float,
        symbols: list[str],
        totalworth: float,
        presetdate: str,
        estimating_startdate: str,
        estimating_enddate: str,
        lamb0: float,
        lamb1: float,
        V: float=10.,
        cacheddir: Union[PathLike, str]=None,
        include_dividends: bool=False
) -> OptimizedPortfolio:
    r, cov = get_BlackScholesMerton_stocks_estimation(
        symbols,
        estimating_startdate,
        estimating_enddate,
        cacheddir=cacheddir,
        include_dividends=include_dividends
    )
    optimized_weighting_policy = OptimizedWeightingPolicyUsingMPTEntropyCostFunction(rf, r, cov, symbols, lamb0, lamb1, V=V)
    optimized_portfolio = OptimizedPortfolio(optimized_weighting_policy, totalworth, presetdate, cacheddir=cacheddir)
    return optimized_portfolio

########### Time-weighted portfolio ##############


def get_exponential_timeweightdf(
        startdate: str,
        enddate: str,
        yearscale: float
) -> pd.DataFrame:
    startdateobj = datetime.strptime(startdate, '%Y-%m-%d')
    enddateobj = datetime.strptime(enddate, '%Y-%m-%d')

    timestamps = np.arange(startdateobj, enddateobj+timedelta(days=1), timedelta(days=1), dtype='datetime64[ns]')
    weights = np.exp(-(len(timestamps)-1-np.arange(len(timestamps)))/365/yearscale)

    timeweightdf = pd.DataFrame({
        'TimeStamp': timestamps,
        'weight': weights
    })
    return timeweightdf


def get_optimized_exponential_timeweighted_portfolio_on_mpt_entropy_costfunction(
        rf: float,
        symbols: list[str],
        totalworth: float,
        presetdate: str,
        estimating_startdate: str,
        estimating_enddate: str,
        yearscale: float,
        lamb0: float,
        lamb1: float,
        V: float=10.,
        cacheddir: Union[PathLike, str]=None,
        include_dividends: bool=False
) -> OptimizedPortfolio:
    timeweightdf = get_exponential_timeweightdf(estimating_startdate, estimating_enddate, yearscale)
    r, cov = get_stocks_timeweighted_estimation(
        symbols,
        timeweightdf,
        cacheddir=cacheddir,
        include_dividends=include_dividends
    )
    optimized_weighting_policy = OptimizedWeightingPolicyUsingMPTEntropyCostFunction(rf, r, cov, symbols, lamb0, lamb1,
                                                                                     V=V)
    optimized_portfolio = OptimizedPortfolio(optimized_weighting_policy, totalworth, presetdate, cacheddir=cacheddir)
    return optimized_portfolio
