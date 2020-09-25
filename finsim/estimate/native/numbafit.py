
import numpy as np
import numba as nb


@nb.njit(nb.types.Tuple((nb.float64, nb.float64))(nb.float64[:], nb.float64[:]))
def numba_fit_BlackScholesMerton_model(ts, prices):
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    r = np.mean(dlogS / dt)
    sigma = np.std(dlogS / np.sqrt(dt))

    return r, sigma


# @nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:, :]))(nb.float64[:], nb.float64[:, :]))
def numba_fit_multivariate_BlackScholesMerton_model(ts, multiprices):
    dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    dt = ts[1:] - ts[:-1]

    r = np.mean(dlogS / dt, axis=1)
    cov = np.cov(dlogS / np.sqrt(dt))

    return r, cov
