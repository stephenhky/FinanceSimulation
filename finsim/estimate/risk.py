
from typing import Literal

import numpy as np
from scipy import stats
from nptyping import NDArray, Shape, Float, Datetime64

from .native.pyrisk import python_estimate_downside_risk, python_estimate_upside_risk
from .constants import dividing_factors_dict


def estimate_downside_risk(
        timestamps: NDArray[Shape["*"], Datetime64],
        prices: NDArray[Shape["*"], Float],
        target_return: float,
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year',
        lowlevellang: Literal['C', 'P']='P'
) -> float:
    """Estimate the downside risk of an asset based on historical price data.
    
    Downside risk measures the volatility of returns that fall below a target return,
    focusing on negative deviations rather than total volatility.
    
    Args:
        timestamps: Array of timestamps corresponding to price observations
        prices: Array of asset prices corresponding to the timestamps
        target_return: The target return threshold for calculating downside risk
        unit: Time unit for calculations. Options are 'second', 'minute', 'hour', 'day', 'year'.
              Default is 'year'.
        lowlevellang: Language for low-level implementation. 'P' for Python, 'C' for Cython.
                       Default is 'P'. Note: Cython implementation is no longer supported.
        
    Returns:
        float: The estimated downside risk (standard deviation of returns below target)
        
    Raises:
        ValueError: If Cython fitting is attempted (no longer supported) or if an unknown
                      low-level language is specified
                      
    Note:
        The function internally converts timestamps to seconds and then to the specified unit.
        The calculation uses the Python implementation of downside risk estimation.
    """
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'P':
        return python_estimate_downside_risk(ts, prices, target_return)
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython))'.format(
                lowlevellang))


def estimate_upside_risk(
        timestamps: NDArray[Shape["*"], Datetime64],
        prices: NDArray[Shape["*"], Float],
        target_return: float,
        unit: Literal['second', 'minute', 'hour', 'day', 'year'] = 'year',
        lowlevellang: Literal['C', 'P'] = 'P'
) -> float:
    """Estimate the upside risk of an asset based on historical price data.
    
    Upside risk measures the volatility of returns that exceed a target return,
    focusing on positive deviations above a minimum acceptable return.
    
    Args:
        timestamps: Array of timestamps corresponding to price observations
        prices: Array of asset prices corresponding to the timestamps
        target_return: The target return threshold for calculating upside risk
        unit: Time unit for calculations. Options are 'second', 'minute', 'hour', 'day', 'year'.
              Default is 'year'.
        lowlevellang: Language for low-level implementation. 'P' for Python, 'C' for Cython.
                       Default is 'P'. Note: Cython implementation is no longer supported.
        
    Returns:
        float: The estimated upside risk (standard deviation of returns above target)
        
    Raises:
        ValueError: If Cython fitting is attempted (no longer supported) or if an unknown
                      low-level language is specified
                      
    Note:
        The function internally converts timestamps to seconds and then to the specified unit.
        The calculation uses the Python implementation of upside risk estimation.
    """
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'P':
        return python_estimate_upside_risk(ts, prices, target_return)
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(
                lowlevellang))


def estimate_beta(
        timestamps: NDArray[Shape["*"], Datetime64],
        prices: NDArray[Shape["*"], Float],
        market_prices: NDArray[Shape["*"], Float],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year'
) -> float:
    """Estimate the beta coefficient of an asset relative to market performance.
    
    Beta measures the sensitivity of an asset's returns to market returns.
    A beta of 1 indicates the asset moves in line with the market, while
    a beta greater than 1 indicates higher volatility than the market.
    
    Args:
        timestamps: Array of timestamps corresponding to price observations
        prices: Array of asset prices corresponding to the timestamps
        market_prices: Array of market index prices corresponding to the timestamps
        unit: Time unit for calculations. Options are 'second', 'minute', 'hour', 'day', 'year'.
              Default is 'year'.
        
    Returns:
        float: The estimated beta coefficient
        
    Note:
        The function uses linear regression to calculate beta, where the slope of the
        regression line represents the beta coefficient.
    """
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor
    dt = ts[1:] - ts[:-1]

    dlogprices = np.log(prices[1:] / prices[:-1])
    dlogmarketprices = np.log(market_prices[1:] / market_prices[:-1])
    assert len(dt) == len(dlogprices)
    assert len(dt) == len(dlogmarketprices)

    stockyields = dlogprices / dt
    marketyields = dlogmarketprices / dt

    reg = stats.linregress(x=marketyields, y=stockyields)
    return reg.slope
