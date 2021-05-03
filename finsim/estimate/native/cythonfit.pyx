
cimport cython
import numpy as np
cimport numpy as np


def cython_fit_BlackScholesMerton_model(np.ndarray[double, ndim=1] ts, np.ndarray[double, ndim=1] prices):
    cdef np.ndarray[double, ndim=1] dlogS = np.log(prices[1:] / prices[:-1])
    cdef np.ndarray[double, ndim=1] dt = ts[1:] - ts[:-1]

    cdef double r = np.mean(dlogS / dt)
    cdef double sigma = np.std(dlogS / np.sqrt(dt))

    return r, sigma


def cython_fit_multivariate_BlackScholesMerton_model(np.ndarray[double, ndim=1] ts, np.ndarray[double, ndim=2] multiprices):
    cdef np.ndarray[double, ndim=2] dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    cdef np.ndarray[double, ndim=1] dt = ts[1:] - ts[:-1]

    cdef np.ndarray[double, ndim=1] r = np.zeros(multiprices.shape[0])
    for i in range(multiprices.shape[0]):
        r[i] = np.mean((dlogS / dt)[i, :])
    cdef np.ndarray[double, ndim=2] cov = np.cov(dlogS / np.sqrt(dt))

    return r, cov
