
import numpy as np
import numba as nb


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64))
def sharpe_ratio(weights, r, cov, rf):
    yieldrate = np.sum(weights * r)
    sqweights = np.expand_dims(weights, axis=1) @ np.expand_dims(weights, axis=0)
    volatility = np.sqrt(np.sum(sqweights * cov))
    return (yieldrate - rf) / volatility


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64, nb.float64, nb.float64))
def mpt_costfunction(weights, r, cov, rf, V0, c):
    weightmat = np.expand_dims(weights[:-1], axis=0)
    return weights[-1]*rf + np.dot(weights[:-1], r) - 0.5*c/V0*(weightmat @ cov @ weightmat.T)[0, 0]
