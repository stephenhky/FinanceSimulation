
from typing import Literal, Annotated

import numpy as np
from numpy.typing import NDArray
import numba as nb


@nb.njit
def python_sharpe_ratio(
        weights: Annotated[NDArray[np.float64], Literal["1D array"]],
        r: Annotated[NDArray[np.float64], Literal["1D array"]],
        cov: Annotated[NDArray[np.float64], Literal["2D array"]],
        rf: float
) -> float:
    """Calculate the Sharpe ratio for a portfolio using Python/Numba.
    
    Args:
        weights: Array of portfolio weights
        r: Array of expected returns
        cov: Covariance matrix
        rf: Risk-free rate
        
    Returns:
        float: Sharpe ratio value
    """
    yieldrate = np.sum(weights * r)
    sqweights = np.expand_dims(weights, axis=1) @ np.expand_dims(weights, axis=0)
    volatility = np.sqrt(np.sum(sqweights * cov))
    return (yieldrate - rf) / volatility


@nb.njit
def python_mpt_costfunction(
        weights: Annotated[NDArray[np.float64], Literal["1D array"]],
        r: Annotated[NDArray[np.float64], Literal["1D array"]],
        cov: Annotated[NDArray[np.float64], Literal["2D array"]],
        rf: float,
        lamb: float,
        V0: float=10.
) -> float:
    """Calculate the MPT cost function for a portfolio using Python/Numba.
    
    Args:
        weights: Array of portfolio weights
        r: Array of expected returns
        cov: Covariance matrix
        rf: Risk-free rate
        lamb: Lambda parameter for the cost function
        V0: Initial portfolio value (default: 10.0)
        
    Returns:
        float: MPT cost function value
    """
    weightmat = np.expand_dims(weights[:-1], axis=0)
    c = lamb * V0
    return weights[-1]*rf + np.dot(weights[:-1], r) - 0.5*c/V0*(weightmat @ cov @ weightmat.T)[0, 0]


@nb.njit
def python_mpt_entropy_costfunction(
        weights: Annotated[NDArray[np.float64], Literal["1D array"]],
        r: Annotated[NDArray[np.float64], Literal["1D array"]],
        cov: Annotated[NDArray[np.float64], Literal["2D array"]],
        rf: float,
        lamb0: float,
        lamb1: float,
        V: float=10.
) -> float:
    """Calculate the MPT entropy cost function for a portfolio using Python/Numba.
    
    Args:
        weights: Array of portfolio weights
        r: Array of expected returns
        cov: Covariance matrix
        rf: Risk-free rate
        lamb0: Lambda 0 parameter for the entropy cost function
        lamb1: Lambda 1 parameter for the entropy cost function
        V: Portfolio value parameter (default: 10.0)
        
    Returns:
        float: MPT entropy cost function value
    """
    weightmat = np.expand_dims(weights[:-1], axis=0)
    c0 = lamb0 * V
    c1 = lamb1 * V
    yield_val = weights[-1]*rf + np.dot(weights[:-1], r)
    cov_val = - 0.5 * c0 / V * (weightmat @ cov @ weightmat.T)[0, 0]
    sumweights = np.sum(weights[:-1])
    entropy_val = - 0.5 * c1 / V * np.sum(weights[:-1] * (np.log(weights[:-1]) - np.log(sumweights))) / sumweights
    return yield_val + cov_val + entropy_val
