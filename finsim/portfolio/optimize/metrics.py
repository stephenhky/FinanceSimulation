
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
    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'P':
        return python_mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=V)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(lowlevellang))

