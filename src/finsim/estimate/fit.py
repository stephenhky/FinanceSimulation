
from typing import Literal, Annotated
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .constants import dividing_factors_dict
from .native.pyfit import python_fit_BlackScholesMerton_model, python_fit_multivariate_BlackScholesMerton_model


# Note: always round-off to seconds first, but flexible about the unit to be used.

def fit_BlackScholesMerton_model(
        timestamps: Annotated[NDArray[np.datetime64], Literal["1D array"]],
        prices: Annotated[NDArray[np.float64], Literal["1D array"]],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year'
) -> tuple[float, float]:
    """Fit a Black-Scholes-Merton model to price data to estimate rate of return and volatility.
    
    This function estimates the parameters of the Black-Scholes-Merton model, which describes
    the dynamics of a financial asset. It calculates the expected rate of return and volatility
    from historical price data.
    
    Args:
        timestamps: Array of timestamps corresponding to price observations
        prices: Array of asset prices corresponding to the timestamps
        unit: Time unit for calculations. Options are 'second', 'minute', 'hour', 'day', 'year'.
              Default is 'year'.

    Returns:
        Tuple containing:
            - rate (float): Estimated rate of return (drift parameter)
            - sigma (float): Estimated volatility (diffusion parameter)
                      
    Note:
        The function internally converts timestamps to seconds and then to the specified unit.
        The calculation uses the Python implementation of the Black-Scholes-Merton model.
    """
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    return python_fit_BlackScholesMerton_model(ts, prices)


def fit_multivariate_BlackScholesMerton_model(
        timestamps: Annotated[NDArray[np.datetime64], Literal["1D array"]],
        multiprices: Annotated[NDArray[np.float64], Literal["1D array"]],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year',
) -> tuple[Annotated[NDArray[np.float64], Literal["1D array"]], Annotated[NDArray[np.float64], Literal["2D array"]]]:
    """Fit a multivariate Black-Scholes-Merton model to price data for multiple assets.
    
    This function estimates the parameters of the multivariate Black-Scholes-Merton model,
    which describes the dynamics of multiple financial assets and their correlations.
    
    Args:
        timestamps: Array of timestamps corresponding to price observations
        multiprices: 2D array of asset prices for multiple assets. Each row represents
                      a different asset, and each column represents prices at a specific time
        unit: Time unit for calculations. Options are 'second', 'minute', 'hour', 'day', 'year'.
              Default is 'year'.

    Returns:
        Tuple containing:
            - rates (NDArray[Shape["*"], Float]): Array of estimated rates of return for each asset
            - covariance_matrix (NDArray[Shape["*, *"], Float]): Estimated covariance matrix of returns

    Note:
        The function internally converts timestamps to seconds and then to the specified unit.
        The calculation uses the Python implementation of the multivariate Black-Scholes-Merton model.
    """
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    return python_fit_multivariate_BlackScholesMerton_model(ts, multiprices)


######## routines below are for time-weighted portfolio building

def fit_timeweighted_BlackScholesMerton_model(
        timestamps: Annotated[NDArray[np.datetime64], Literal["1D array"]],
        prices: Annotated[NDArray[np.float64], Literal["1D array"]],
        weights: Annotated[NDArray[np.float64], Literal["1D array"]],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year'
) -> tuple[float, float]:
    """Fit a time-weighted Black-Scholes-Merton model to price data.
    
    This function estimates the parameters of the Black-Scholes-Merton model using
    time-weighted observations, giving different importance to different time periods.
    
    Args:
        timestamps: Array of timestamps corresponding to price observations
        prices: Array of asset prices corresponding to the timestamps
        weights: Array of weights for time-weighted calculations (same length as timestamps)
        unit: Time unit for calculations. Options are 'second', 'minute', 'hour', 'day', 'year'.
              Default is 'year'.
        
    Returns:
        Tuple containing:
            - rate (float): Time-weighted estimated rate of return
            - sigma (float): Time-weighted estimated volatility
    """
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    r = np.average(dlogS / dt, weights=weights[1:])
    sigma = np.sqrt(np.average(np.square(dlogS / np.sqrt(dt)), weights=weights[1:]) - np.square(
        np.average(dlogS / np.sqrt(dt), weights=weights[1:])))

    return r, sigma


def fit_timeweighted_multivariate_BlackScholesMerton_model(
        timestamps: Annotated[NDArray[np.datetime64], Literal["1D array"]],
        multiprices: Annotated[NDArray[np.float64], Literal["2D array"]],
        weights: Annotated[NDArray[np.float64], Literal["1D array"]],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year'
) -> tuple[Annotated[NDArray[np.float64], Literal["1D array"]], Annotated[NDArray[np.float64], Literal["2D array"]]]:
    """Fit a time-weighted multivariate Black-Scholes-Merton model to price data for multiple assets.
    
    This function estimates the parameters of the multivariate Black-Scholes-Merton model
    using time-weighted observations for multiple assets.
    
    Args:
        timestamps: Array of timestamps corresponding to price observations
        multiprices: 2D array of asset prices for multiple assets. Each row represents
                      a different asset, and each column represents prices at a specific time
        weights: Array of weights for time-weighted calculations (same length as timestamps)
        unit: Time unit for calculations. Options are 'second', 'minute', 'hour', 'day', 'year'.
              Default is 'year'.
        
    Returns:
        Tuple containing:
            - rates (NDArray[Shape["*"], Float]): Array of time-weighted estimated rates of return
            - covariance_matrix (NDArray[Shape["*, *"], Float]): Time-weighted estimated covariance matrix
    """
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    # possibly hit this part
    dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    dt = ts[1:] - ts[:-1]

    # estimation
    r = np.zeros(multiprices.shape[0])
    for i in range(multiprices.shape[0]):
        r[i] = np.average((dlogS[i, :] / dt), weights=weights[1:])
    cov = np.zeros((multiprices.shape[0], multiprices.shape[0]))
    for i, j in product(range(multiprices.shape[0]), range(multiprices.shape[0])):
        avg_i = np.average(dlogS[i, :] / np.sqrt(dt), weights=weights[1:])
        avg_j = np.average(dlogS[j, :] / np.sqrt(dt), weights=weights[1:])
        cov[i, j] = np.average(dlogS[i, :] * dlogS[j, :] / dt, weights=np.square(weights[1:])) - avg_i * avg_j

    return r, cov
