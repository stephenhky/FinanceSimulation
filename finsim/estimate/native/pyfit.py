
from typing import Tuple

import numpy as np
import numba as nb
from nptyping import NDArray, Shape, Float


@nb.njit
def python_fit_BlackScholesMerton_model(
        ts: NDArray[Shape["*"], Float],
        prices: NDArray[Shape["*"], Float]
) -> Tuple[float, float]:
    """Fit a Black-Scholes-Merton model to price data using Python implementation.
    
    This function estimates the parameters of the Black-Scholes-Merton model, which describes
    the dynamics of a financial asset. It calculates the expected rate of return and volatility
    from historical price data.
    
    Args:
        ts: Array of time values (converted to specified unit)
        prices: Array of asset prices corresponding to the time values
        
    Returns:
        Tuple containing:
            - r (float): Estimated rate of return (drift parameter)
            - sigma (float): Estimated volatility (diffusion parameter)
    """
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    r = np.mean(dlogS / dt)
    sigma = np.std(dlogS / np.sqrt(dt))

    return r, sigma


@nb.njit
def python_fit_multivariate_BlackScholesMerton_model(
        ts: NDArray[Shape["*"], Float],
        multiprices: NDArray[Shape["*, *"], Float]
) -> Tuple[NDArray[Shape["*"], Float], NDArray[Shape["*, *"], Float]]:
    """Fit a multivariate Black-Scholes-Merton model to price data for multiple assets.
    
    This function estimates the parameters of the multivariate Black-Scholes-Merton model,
    which describes the dynamics of multiple financial assets and their correlations.
    
    Args:
        ts: Array of time values (converted to specified unit)
        multiprices: 2D array of asset prices for multiple assets. Each row represents
                      a different asset, and each column represents prices at a specific time
        
    Returns:
        Tuple containing:
            - r (NDArray[Shape["*"], Float]): Array of estimated rates of return for each asset
            - cov (NDArray[Shape["*, *"], Float]): Estimated covariance matrix of returns
    """
    dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    dt = ts[1:] - ts[:-1]

    r = np.zeros(multiprices.shape[0])
    for i in range(multiprices.shape[0]):
        r[i] = np.mean((dlogS[i, :] / dt))
    cov = np.cov(dlogS / np.sqrt(dt), bias=True)

    return r, cov
