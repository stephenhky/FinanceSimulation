
cimport cython
import numpy as np
cimport numpy as np

from libc.math cimport log, sqrt


def cython_fit_BlackScholesMerton_model(np.ndarray[double, ndim=1] ts, np.ndarray[double, ndim=1] prices):
    cdef int nbpts = len(prices)
    cdef int i
    cdef double dlogS, dt

    cdef double sumpr = 0.
    cdef double sumnoise = 0.
    cdef double sumsqnoise = 0.

    for i in range(nbpts-1):
        dlogS = log(prices[i+1]/prices[i])
        dt = ts[i+1] - ts[i]

        sumpr += dlogS / dt
        sumnoise += dlogS / sqrt(dt)
        sumsqnoise += dlogS * dlogS / dt

    cdef double r = sumpr / (nbpts - 1)
    cdef double sigma = sqrt(sumsqnoise / (nbpts-1) - sumnoise*sumnoise/(nbpts-1)/(nbpts-1))

    return r, sigma


def cython_fit_multivariate_BlackScholesMerton_model(np.ndarray[double, ndim=1] ts, np.ndarray[double, ndim=2] multiprices):
    cdef np.ndarray[double, ndim=2] dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    cdef np.ndarray[double, ndim=1] dt = ts[1:] - ts[:-1]

    cdef np.ndarray[double, ndim=1] r = np.zeros(multiprices.shape[0])
    for i in range(multiprices.shape[0]):
        r[i] = np.mean((dlogS / dt)[i, :])
    cdef np.ndarray[double, ndim=2] cov = np.cov(dlogS / np.sqrt(dt), bias=True)

    return r, cov
