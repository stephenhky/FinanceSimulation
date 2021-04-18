
import numpy as np
cimport numpy as np


def cython_sharpe_ratio(np.ndarray weights, np.ndarray r, np.ndarray cov, double rf):
    cdef double yieldrate = np.sum(weights * r)
    cdef np.ndarray sqweights = np.expand_dims(weights, axis=1) @ np.expand_dims(weights, axis=0)
    cdef double volatility = np.sqrt(np.sum(sqweights * cov))
    return (yieldrate - rf) / volatility
