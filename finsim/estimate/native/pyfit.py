
from typing import Tuple

import numpy as np
import numpy.typing as npt
import numba as nb


@nb.njit
def python_fit_BlackScholesMerton_model(
        ts: npt.NDArray[np.float64],
        prices: npt.NDArray[np.float64]
) -> Tuple[float, float]:
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    r = np.mean(dlogS / dt)
    sigma = np.std(dlogS / np.sqrt(dt))

    return r, sigma


@nb.njit
def python_fit_multivariate_BlackScholesMerton_model(
        ts: npt.NDArray[np.float64],
        multiprices: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    dt = ts[1:] - ts[:-1]

    r = np.zeros(multiprices.shape[0])
    for i in range(multiprices.shape[0]):
        r[i] = np.mean((dlogS[i, :] / dt))
    cov = np.cov(dlogS / np.sqrt(dt), bias=True)

    return r, cov
