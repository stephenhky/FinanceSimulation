
import numpy as np
import numba as nb


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64))
def numba_sharpe_ratio(weights, r, cov, rf):
    yieldrate = np.sum(weights * r)
    sqweights = np.expand_dims(weights, axis=1) @ np.expand_dims(weights, axis=0)
    volatility = np.sqrt(np.sum(sqweights * cov))
    return (yieldrate - rf) / volatility


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64, nb.float64, nb.float64))
def numba_mpt_costfunction(weights, r, cov, rf, lamb, V0=10.):
    weightmat = np.expand_dims(weights[:-1], axis=0)
    c = lamb * V0
    return weights[-1]*rf + np.dot(weights[:-1], r) - 0.5*c/V0*(weightmat @ cov @ weightmat.T)[0, 0]


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64, nb.float64, nb.float64, nb.float64))
def numba_mpt_entropy_costfunction(weights, r, cov, rf, lamb0, lamb1, V=10.):
    weightmat = np.expand_dims(weights[:-1], axis=0)
    c0 = lamb0 * V
    c1 = lamb1 * V
    yield_val = weights[-1]*rf + np.dot(weights[:-1], r)
    cov_val = - 0.5 * c0 / V * (weightmat @ cov @ weightmat.T)[0, 0]
    sumweights = np.sum(weights[:-1])
    entropy_val = - 0.5 * c1 / V * np.sum(weights[:-1] * (np.log(weights[-1]) - sumweights)) / sumweights
    return yield_val + cov_val + entropy_val
