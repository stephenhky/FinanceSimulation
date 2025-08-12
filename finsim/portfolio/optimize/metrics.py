
from typing import Literal

import numpy as np
from nptyping import NDArray, Shape, Float

from .native.pymetrics import python_sharpe_ratio, python_mpt_costfunction, python_mpt_entropy_costfunction


def sharpe_ratio(
        weights: NDArray[Shape["*"], Float],
        r: NDArray[Shape["*"], Float],
        cov: NDArray[Shape["*, *"], Float],
        rf: float,
        lowlevellang: Literal['C', 'P']='P'
) -> float:
    """Calculate the Sharpe ratio for a portfolio.
    
    Args:
        weights: Array of portfolio weights
        r: Array of expected returns
        cov: Covariance matrix
        rf: Risk-free rate
        lowlevellang: Language for low-level implementation, 'P' for Python (default: 'P')
        
    Returns:
        float: Sharpe ratio value
        
    Raises:
        ValueError: If Cython fitting is attempted (no longer supported)
    """
    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'P':
        return python_sharpe_ratio(weights, r, cov, rf)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(lowlevellang))


def mpt_costfunction(
        weights: NDArray[Shape["*"], Float],
        r: NDArray[Shape["*"], Float],
        cov: NDArray[Shape["*, *"], Float],
        rf: float,
        lamb: float,
        V0: float=10.,
        lowlevellang: Literal['C', 'P']='P'
) -> float:
    """Calculate the MPT cost function for a portfolio.
    
    Args:
        weights: Array of portfolio weights
        r: Array of expected returns
        cov: Covariance matrix
        rf: Risk-free rate
        lamb: Lambda parameter for the cost function
        V0: Initial portfolio value (default: 10.0)
        lowlevellang: Language for low-level implementation, 'P' for Python (default: 'P')
        
    Returns:
        float: MPT cost function value
        
    Raises:
        ValueError: If Cython fitting is attempted (no longer supported)
    """
    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'P':
        return python_mpt_costfunction(weights, r, cov, rf, lamb, V0=V0)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(lowlevellang))


def mpt_entropy_costfunction(
        weights: NDArray[Shape["*"], Float],
        r: NDArray[Shape["*"], Float],
        cov: NDArray[Shape["*, *"], Float],
        rf: float,
        lamb0: float,
        lamb1: float,
        V: float=10.,
        lowlevellang: Literal['C', 'P']='C'
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
        lowlevellang: Language for low-level implementation, 'P' for Python (default: 'P')
        
    Returns:
        float: MPT entropy cost function value
        
    Raises:
        ValueError: If Cython fitting is attempted (no longer supported)
    """
    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'P':
        return python_mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=V)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(lowlevellang))

