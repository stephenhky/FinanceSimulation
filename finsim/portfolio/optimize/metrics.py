
from .native.pymetrics import python_sharpe_ratio, python_mpt_costfunction, python_mpt_entropy_costfunction
from .native.cythonmetrics import cython_sharpe_ratio, cython_mpt_costfunction, cython_mpt_entropy_costfunction


def sharpe_ratio(weights, r, cov, rf, lowlevellang='C'):
    if lowlevellang == 'C':
        return cython_sharpe_ratio(weights, r, cov, rf)
    elif lowlevellang == 'P':
        return python_sharpe_ratio(weights, r, cov, rf)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(lowlevellang))


def mpt_costfunction(weights, r, cov, rf, lamb, V0=10., lowlevellang='C'):
    if lowlevellang == 'C':
        return cython_mpt_costfunction(weights, r, cov, rf, lamb, V0)
    elif lowlevellang == 'P':
        return python_mpt_costfunction(weights, r, cov, rf, lamb, V0=V0)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(lowlevellang))


def mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=10., lowlevellang='C'):
    if lowlevellang == 'C':
        return cython_mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V)
    elif lowlevellang == 'P':
        return python_mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=V)
    else:
        raise ValueError('Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(lowlevellang))

