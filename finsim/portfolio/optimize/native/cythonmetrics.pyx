
import numpy as np
cimport numpy as np


def cython_sharpe_ratio(np.ndarray weights, np.ndarray r, np.ndarray cov, double rf):
    cdef double yieldrate = np.sum(weights * r)
    cdef np.ndarray sqweights = np.expand_dims(weights, axis=1) @ np.expand_dims(weights, axis=0)
    cdef double volatility = np.sqrt(np.sum(sqweights * cov))
    return (yieldrate - rf) / volatility


def cython_mpt_costfunction(np.ndarray weights, np.ndarray r, np.ndarray cov, double rf, double lamb, double V0):
    cdef np.ndarray weightmat = np.expand_dims(weights[:-1], axis=0)
    cdef double c = lamb * V0
    cdef double cost = weights[-1]*rf + np.dot(weights[:-1], r) - 0.5*c/V0*(weightmat @ cov @ weightmat.T)[0, 0]
    return cost


def cython_mpt_entropy_costfunction(np.ndarray weights, np.ndarray r, np.ndarray cov, double rf, double lamb0, double lamb1, double V):
    cdef np.ndarray weightmat = np.expand_dims(weights[:-1], axis=0)
    cdef double c0 = lamb0 * V
    cdef double c1 = lamb1 * V
    cdef double yield_val = weights[-1]*rf + np.dot(weights[:-1], r)
    cdef double cov_val = - 0.5 * c0 / V * (weightmat @ cov @ weightmat.T)[0, 0]
    cdef double sumweights = np.sum(weights[:-1])
    cdef double entropy_val = - 0.5 * c1 / V * np.sum(weights[:-1] * (np.log(weights[-1]) - sumweights)) / sumweights
    return yield_val + cov_val + entropy_val
