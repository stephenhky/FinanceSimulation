
from .native.numbametrics import numba_sharpe_ratio, numba_mpt_costfunction, numba_mpt_entropy_costfunction


def sharpe_ratio(weights, r, cov, rf):
    return numba_sharpe_ratio(weights, r, cov, rf)


def mpt_costfunction(weights, r, cov, rf, lamb, V0=10.):
    return numba_mpt_costfunction(weights, r, cov, rf, lamb, V0=V0)


def mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=10.):
    return numba_mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=V)
