
cimport cython
import numpy as np
cimport numpy as np


def cython_sharpe_ratio(np.ndarray[np.float64_t, ndim=1] weights, np.ndarray[np.float64_t, ndim=1] r, np.ndarray[np.float64_t, ndim=2] cov, float rf):
    cdef float yieldrate = np.sum(weights * r)
    cdef np.ndarray[np.float64_t, ndim=2] sqweights = np.expand_dims(weights, axis=1) @ np.expand_dims(weights, axis=0)
    cdef float volatility = np.sqrt(np.sum(sqweights * cov))
    return (yieldrate - rf) / volatility


def cython_mpt_costfunction(np.ndarray[np.float64_t, ndim=1] weights, np.ndarray[np.float64_t, ndim=1] r, np.ndarray[np.float64_t, ndim=2] cov, float rf, float lamb, float V0):
    cdef np.ndarray[np.float64_t, ndim=2] weightmat = np.expand_dims(weights[:-1], axis=0)
    cdef float c = lamb * V0
    cdef float cost = weights[-1]*rf + np.dot(weights[:-1], r) - 0.5*c/V0*(weightmat @ cov @ weightmat.T)[0, 0]
    return cost


def cython_mpt_entropy_costfunction(np.ndarray[np.float64_t, ndim=1] weights, np.ndarray[np.float64_t, ndim=1] r, np.ndarray[np.float64_t, ndim=2] cov, float rf, float lamb0, float lamb1, float V):
    cdef np.ndarray[np.float64_t, ndim=2] weightmat = np.expand_dims(weights[:-1], axis=0)
    cdef float c0 = lamb0 * V
    cdef float c1 = lamb1 * V
    cdef float yield_val = weights[-1]*rf + np.dot(weights[:-1], r)
    cdef float cov_val = - 0.5 * c0 / V * (weightmat @ cov @ weightmat.T)[0, 0]
    cdef float sumweights = np.sum(weights[:-1])
    cdef float entropy_val = - 0.5 * c1 / V * np.sum(weights[:-1] * (np.log(weights[:-1]) - np.log(sumweights))) / sumweights
    return yield_val + cov_val + entropy_val
