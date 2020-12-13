
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
    weightmat = np.expand_dims(weights[:-1], axis=0)
    return weights[-1]*rf + np.dot(weights[:-1], r) - 0.5*c/V0*np.matmul(np.matmul(weightmat, cov), weightmat.T)[0, 0]
