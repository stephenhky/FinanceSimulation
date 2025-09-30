
from typing import Literal, Annotated

import numpy as np
from numpy.typing import NDArray
import numba as nb


@nb.njit
def python_estimate_downside_risk(
        ts: Annotated[NDArray[np.float64], Literal["1D array"]],
        prices: Annotated[NDArray[np.float64], Literal["1D array"]],
        target_return: float
) -> float:
    """Estimate the downside risk of an asset based on historical price data.
    
    Downside risk measures the volatility of returns that fall below a target return,
    focusing on negative deviations rather than total volatility.
    
    Args:
        ts: Array of time values (converted to specified unit)
        prices: Array of asset prices corresponding to the time values
        target_return: The target return threshold for calculating downside risk
        
    Returns:
        float: The estimated downside risk (standard deviation of returns below target)
    """
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    rms_rarray = dlogS / np.sqrt(dt)
    less_return_array = target_return - rms_rarray
    less_return_array[less_return_array < 0] = 0.
    downside_risk = np.sqrt(np.mean(np.square(less_return_array)))
    return downside_risk


@nb.njit
def python_estimate_upside_risk(
        ts: Annotated[NDArray[np.float64], Literal["1D array"]],
        prices: Annotated[NDArray[np.float64], Literal["1D array"]],
        target_return: float
) -> float:
    """Estimate the upside risk of an asset based on historical price data.
    
    Upside risk measures the volatility of returns that exceed a target return,
    focusing on positive deviations above a minimum acceptable return.
    
    Args:
        ts: Array of time values (converted to specified unit)
        prices: Array of asset prices corresponding to the time values
        target_return: The target return threshold for calculating upside risk
        
    Returns:
        float: The estimated upside risk (standard deviation of returns above target)
    """
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    rms_rarray = dlogS / np.sqrt(dt)
    more_return_array = rms_rarray - target_return
    more_return_array[more_return_array < 0] = 0.
    upside_risk = np.sqrt(np.mean(np.square(more_return_array)))
    return upside_risk
