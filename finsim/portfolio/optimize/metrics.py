
from .native.numbametrics import numba_sharpe_ratio, numba_mpt_costfunction, numba_mpt_entropy_costfunction
from .native.cythonmetrics import cython_sharpe_ratio, cython_mpt_costfunction, cython_mpt_entropy_costfunction


def sharpe_ratio(weights, r, cov, rf, lowlevellang='C'):
    if lowlevellang == 'C':
        return cython_sharpe_ratio(weights, r, cov, rf)
    elif lowlevellang == 'N':
        return numba_sharpe_ratio(weights, r, cov, rf)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "N" (numba), "C" (Cython), or "F" (Fortran).)'.format(lowlevellang))


def mpt_costfunction(weights, r, cov, rf, lamb, V0=10., lowlevellang='C'):
    if lowlevellang == 'C':
        return cython_mpt_costfunction(weights, r, cov, rf, lamb, V0)
    elif lowlevellang == 'N':
        return numba_mpt_costfunction(weights, r, cov, rf, lamb, V0=V0)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "N" (numba), "C" (Cython), or "F" (Fortran).)'.format(lowlevellang))


def mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=10., lowlevellang='C'):
    if lowlevellang == 'C':
        return cython_mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V)
    elif lowlevellang == 'N':
        return numba_mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=V)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "N" (numba), "C" (Cython), or "F" (Fortran).)'.format(lowlevellang))

