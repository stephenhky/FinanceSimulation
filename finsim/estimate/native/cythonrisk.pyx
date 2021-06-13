
import numpy as np
cimport numpy as np

from libc.math cimport log, sqrt


def cython_estimate_downside_risk(np.ndarray[double, ndim=1] ts, np.ndarray[double, ndim=1] prices, double target_return):
    cdef int nbpts = len(prices)
    cdef double logS, dt, rms, sum_down_sqrms
    cdef int i

    sum_down_sqrms = 0.
    for i in range(nbpts-1):
        dlogS = log(prices[i+1] / prices[i])
        dt = ts[i+1] - ts[i]
        rms = dlogS / sqrt(dt)
        if rms < target_return:
            sum_down_sqrms += (target_return-rms) * (target_return-rms)

    cdef downside_risk = sqrt(sum_down_sqrms / (nbpts-1))

    return downside_risk


def cython_estimate_upside_risk(np.ndarray[double, ndim=1] ts, np.ndarray[double, ndim=1] prices, double target_return):
    cdef int nbpts = len(prices)
    cdef double logS, dt, rms, sum_up_sqrms
    cdef int i

    sum_up_sqrms = 0.
    for i in range(nbpts-1):
        dlogS = log(prices[i+1] / prices[i])
        dt = ts[i+1] - ts[i]
        rms = dlogS / sqrt(dt)
        if rms > target_return:
            sum_up_sqrms += (rms-target_return) * (rms-target_return)

    cdef upside_risk = sqrt(sum_up_sqrms / (nbpts-1))

    return upside_risk
