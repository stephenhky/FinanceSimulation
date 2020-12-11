
import numpy as np


def sharpe_ratio(weights, r, cov, rf):
    yieldrate = np.sum(weights * r)
    sqweights = np.matmul(
        np.expand_dims(weights, axis=1),
        np.expand_dims(weights, axis=0)
    )
    volatility = np.sqrt(np.sum(sqweights * cov))
    return (yieldrate - rf) / volatility


def mpt_costfunction(weights, r, cov, rf, V0, c):
    rmat = np.expand_dims(r, axis=0)
    return weights[-1]*rf + weights*r - 0.5*c/V0*np.matmul(np.matmul(rmat, cov), rmat.T)
