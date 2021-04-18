
import numpy as np
cimport numpy as np


def cython_fit_BlackScholesMerton_model(np.ndarray ts, np.ndarray prices):
    cdef np.ndarray dlogS = np.log(prices[1:] / prices[:-1])
    cdef np.ndarray dt = ts[1:] - ts[:-1]

    cdef double r = np.mean(dlogS / dt)
    cdef double sigma = np.std(dlogS / np.sqrt(dt))

    return r, sigma


def cython_fit_multivariate_BlackScholesMerton_model(np.ndarray ts, np.ndarray multiprices):
    cdef np.ndarray dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    cdef np.ndarray dt = ts[1:] - ts[:-1]

    cdef np.ndarray r = np.zeros(multiprices.shape[0])
    for i in range(multiprices.shape[0]):
        r[i] = np.mean((dlogS / dt)[i, :])
    cdef np.ndarray cov = np.cov(dlogS / np.sqrt(dt))

    return r, cov
