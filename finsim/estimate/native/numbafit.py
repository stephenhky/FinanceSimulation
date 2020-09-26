
import numpy as np
import numba as nb


@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64[:], nb.float64[:]))
def numba_fit_BlackScholesMerton_model(ts, prices):
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    r = np.mean(dlogS / dt)
    sigma = np.std(dlogS / np.sqrt(dt))

    return r, sigma


@nb.njit(nb.types.Tuple((nb.float64[:], nb.float64[:, :]))(nb.float64[:], nb.float64[:, :]))
def numba_fit_multivariate_BlackScholesMerton_model(ts, multiprices):
    dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    dt = ts[1:] - ts[:-1]

    r = np.zeros(multiprices.shape[0])
    for i in range(multiprices.shape[0]):
        r[i] = np.mean((dlogS / dt)[i, :])
    # r = np.mean((dlogS / dt), axis=1)
    cov = np.cov(dlogS / np.sqrt(dt))

    return r, cov
