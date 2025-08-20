
from datetime import datetime, timedelta
import warnings
from os import PathLike
from typing import Optional

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
        cacheddir: Optional[PathLike | str]=None,
        include_dividends: bool=False
) -> OptimizedPortfolio:
    """Create an optimized portfolio based on the Sharpe ratio optimization method.
    
    This function creates a portfolio optimized using the Sharpe ratio maximization approach
    from Modern Portfolio Theory (MPT). It estimates asset returns and covariances using
    the Black-Scholes-Merton model and then optimizes weights to maximize the Sharpe ratio.
    
    Args:
        rf: Risk-free rate
        symbols: List of stock symbols to include in the portfolio
        totalworth: Total value of the portfolio
        presetdate: Date for which to calculate the portfolio composition
        estimating_startdate: Start date for return estimation
        estimating_enddate: End date for return estimation
        minweight: Minimum weight for each asset (default: 0.0)
        lazy: Deprecated parameter (default: False)
        cacheddir: Directory for cached data (optional)
        include_dividends: Whether to include dividends in calculations (default: False)
        
    Returns:
        OptimizedPortfolio: An optimized portfolio object with calculated weights
    """
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
        cacheddir: Optional[PathLike | str]=None,
        include_dividends: bool=False
) -> OptimizedPortfolio:
    """Create an optimized portfolio based on the MPT cost function optimization method.
    
    This function creates a portfolio optimized using a cost function approach from
    Modern Portfolio Theory (MPT). It estimates asset returns and covariances using
    the Black-Scholes-Merton model and then optimizes weights using a cost function.
    
    Args:
        rf: Risk-free rate
        symbols: List of stock symbols to include in the portfolio
        totalworth: Total value of the portfolio
        presetdate: Date for which to calculate the portfolio composition
        estimating_startdate: Start date for return estimation
        estimating_enddate: End date for return estimation
        lamb: Lambda parameter for the cost function
        V0: Initial portfolio value (default: 10.0)
        cacheddir: Directory for cached data (optional)
        include_dividends: Whether to include dividends in calculations (default: False)
        
    Returns:
        OptimizedPortfolio: An optimized portfolio object with calculated weights
    """
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
        cacheddir: Optional[PathLike | str]=None,
        include_dividends: bool=False
) -> OptimizedPortfolio:
    """Create an optimized portfolio based on the MPT entropy cost function optimization method.
    
    This function creates a portfolio optimized using an entropy cost function approach
    from Modern Portfolio Theory (MPT). It estimates asset returns and covariances using
    the Black-Scholes-Merton model and then optimizes weights using an entropy cost function.
    
    Args:
        rf: Risk-free rate
        symbols: List of stock symbols to include in the portfolio
        totalworth: Total value of the portfolio
        presetdate: Date for which to calculate the portfolio composition
        estimating_startdate: Start date for return estimation
        estimating_enddate: End date for return estimation
        lamb0: Lambda 0 parameter for the entropy cost function
        lamb1: Lambda 1 parameter for the entropy cost function
        V: Portfolio value parameter (default: 10.0)
        cacheddir: Directory for cached data (optional)
        include_dividends: Whether to include dividends in calculations (default: False)
        
    Returns:
        OptimizedPortfolio: An optimized portfolio object with calculated weights
    """
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
    """Generate exponential time weights for portfolio optimization.
    
    This function creates a DataFrame with exponentially decaying weights over time,
    which can be used for time-weighted portfolio optimization.
    
    Args:
        startdate: Start date in 'YYYY-MM-DD' format
        enddate: End date in 'YYYY-MM-DD' format
        yearscale: Time scale parameter for exponential decay
        
    Returns:
        pd.DataFrame: DataFrame with 'TimeStamp' and 'weight' columns
    """
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
        cacheddir: Optional[PathLike | str]=None,
        include_dividends: bool=False
) -> OptimizedPortfolio:
    """Create an optimized time-weighted portfolio using exponential weights and entropy cost function.
    
    This function creates a portfolio optimized using an entropy cost function approach
    from Modern Portfolio Theory (MPT) with exponential time weighting. It estimates
    asset returns and covariances using the Black-Scholes-Merton model with time weights
    and then optimizes weights using an entropy cost function.
    
    Args:
        rf: Risk-free rate
        symbols: List of stock symbols to include in the portfolio
        totalworth: Total value of the portfolio
        presetdate: Date for which to calculate the portfolio composition
        estimating_startdate: Start date for return estimation
        estimating_enddate: End date for return estimation
        yearscale: Time scale parameter for exponential decay
        lamb0: Lambda 0 parameter for the entropy cost function
        lamb1: Lambda 1 parameter for the entropy cost function
        V: Portfolio value parameter (default: 10.0)
        cacheddir: Directory for cached data (optional)
        include_dividends: Whether to include dividends in calculations (default: False)
        
    Returns:
        OptimizedPortfolio: An optimized time-weighted portfolio object with calculated weights
    """
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
