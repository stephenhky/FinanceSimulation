
from typing import Literal, Annotated

import numpy as np
from numpy.typing import NDArray

from .native.pymetrics import python_sharpe_ratio, python_mpt_costfunction, python_mpt_entropy_costfunction


def sharpe_ratio(
        weights: Annotated[NDArray[np.float64], Literal["1D array"]],
        r: Annotated[NDArray[np.float64], Literal["1D array"]],
        cov: Annotated[NDArray[np.float64], Literal["2D array"]],
        rf: float
) -> float:
    """Calculate the Sharpe ratio for a portfolio.
    
    Args:
        weights: Array of portfolio weights
        r: Array of expected returns
        cov: Covariance matrix
        rf: Risk-free rate
        
    Returns:
        float: Sharpe ratio value
    """
    return python_sharpe_ratio(weights, r, cov, rf)


def mpt_costfunction(
        weights: Annotated[NDArray[np.float64], Literal["1D array"]],
        r: Annotated[NDArray[np.float64], Literal["1D array"]],
        cov: Annotated[NDArray[np.float64], Literal["2D array"]],
        rf: float,
        lamb: float,
        V0: float=10.
) -> float:
    """Calculate the MPT cost function for a portfolio.
    
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
    return python_mpt_costfunction(weights, r, cov, rf, lamb, V0=V0)


def mpt_entropy_costfunction(
        weights: Annotated[NDArray[np.float64], Literal["1D array"]],
        r: Annotated[NDArray[np.float64], Literal["1D array"]],
        cov: Annotated[NDArray[np.float64], Literal["2D array"]],
        rf: float,
        lamb0: float,
        lamb1: float,
        V: float=10.
) -> float:
    """Calculate the MPT entropy cost function for a portfolio.
    
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
        
    Raises:
        ValueError: If Cython fitting is attempted (no longer supported)
    """
    return python_mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=V)
