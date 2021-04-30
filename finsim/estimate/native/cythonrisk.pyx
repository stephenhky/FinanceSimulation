
import numpy as np
cimport numpy as np


def cython_estimate_downside_risk(np.ndarray[double, ndim=1] ts, np.ndarray[double, ndim=1] prices, double target_return):
    cdef np.ndarray[double, ndim=1] dlogS = np.log(prices[1:] / prices[:-1])
    cdef np.ndarray[double, ndim=1] dt = ts[1:] - ts[:-1]

    cdef np.ndarray[double, ndim=1] rms_rarray = dlogS / np.sqrt(dt)
    cdef np.ndarray[double, ndim=1] less_return_array = target_return - rms_rarray
    less_return_array[less_return_array < 0] = 0.
    cdef double downside_risk = np.sqrt(np.mean(np.square(less_return_array)))
    return downside_risk


def cython_estimate_upside_risk(np.ndarray[double, ndim=1] ts, np.ndarray[double, ndim=1] prices, double target_return):
    cdef np.ndarray[double, ndim=1] dlogS = np.log(prices[1:] / prices[:-1])
    cdef np.ndarray[double, ndim=1] dt = ts[1:] - ts[:-1]

    cdef np.ndarray[double, ndim=1] rms_rarray = dlogS / np.sqrt(dt)
    cdef np.ndarray[double, ndim=1] more_return_array = rms_rarray - target_return
    more_return_array[more_return_array < 0] = 0.
    cdef double upside_risk = np.sqrt(np.mean(np.square(more_return_array)))
    return upside_risk
