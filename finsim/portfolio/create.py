
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd

from .optimize.numerics import get_BlackScholesMerton_stocks_estimation, get_stocks_timeweighted_estimation
from .portfolio import OptimizedPortfolio
from .optimize.policy import OptimizedWeightingPolicyUsingMPTSharpeRatio, \
    OptimizedWeightingPolicyUsingMPTCostFunction, OptimizedWeightingPolicyUsingMPTEntropyCostFunction


def get_optimized_portfolio_on_sharpe_ratio(
        rf,
        symbols,
        totalworth,
        presetdate,
        estimating_startdate,
        estimating_enddate,
        minweight=0.,
        lazy=False,
        cacheddir=None,
        include_dividends=False
):
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
        rf,
        symbols,
        totalworth,
        presetdate,
        estimating_startdate,
        estimating_enddate,
        lamb,
        V0=10.,
        cacheddir=None,
        include_dividends=False
):
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
        rf,
        symbols,
        totalworth,
        presetdate,
        estimating_startdate,
        estimating_enddate,
        lamb0,
        lamb1,
        V=10.,
        cacheddir=None,
        include_dividends=False
):
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


def get_exponential_timeweightdf(startdate, enddate, yearscale):
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
        rf,
        symbols,
        totalworth,
        presetdate,
        estimating_startdate,
        estimating_enddate,
        yearscale,
        lamb0,
        lamb1,
        V=10.,
        cacheddir=None,
        include_dividends=False
):
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
