
import numpy as np
import numba as nb


@nb.njit(nb.types.Tuple(nb.float64, nb.float64)(nb.datetime[:], nb.float64[:], nb.int64))
def numba_fit_BlackScholesMerton_model(timestamps, prices, dividing_factor):
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = np.array(timestamps[1:] - timestamps[:-1], dtype='timedelta64[s]')
    dt = np.array(dt, dtype=np.float) / dividing_factor

    r = np.mean(dlogS / dt)
    sigma = np.std(dlogS / np.sqrt(dt))

    return r, sigma


@nb.njit(nb.types.Tuple(nb.float64[:], nb.float64[:, :])(nb.datetime[:], nb.float64[:, :], nb.int64))
def numba_fit_multivariate_BlackScholesMerton_model(timestamps, multiprices, dividing_factor):
    dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    dt = np.array(timestamps[1:] - timestamps[:-1], dtype='timedelta64[s]')
    dt = np.array(dt, dtype=np.float) / dividing_factor

    r = np.mean(dlogS / dt, axis=1)
    cov = np.cov(dlogS / np.sqrt(dt))

    return r, cov