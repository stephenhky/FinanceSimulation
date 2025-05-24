
from typing import Tuple

import numpy as np
import numba as nb
from nptyping import NDArray, Shape, Float


@nb.njit
def python_fit_BlackScholesMerton_model(
        ts: NDArray[Shape["*"], Float],
        prices: NDArray[Shape["*"], Float]
) -> Tuple[float, float]:
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    r = np.mean(dlogS / dt)
    sigma = np.std(dlogS / np.sqrt(dt))

    return r, sigma


@nb.njit
def python_fit_multivariate_BlackScholesMerton_model(
        ts: NDArray[Shape["*"], Float],
        multiprices: NDArray[Shape["*, *"], Float]
) -> Tuple[NDArray[Shape["*"], Float], NDArray[Shape["*, *"], Float]]:
    dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    dt = ts[1:] - ts[:-1]

    r = np.zeros(multiprices.shape[0])
    for i in range(multiprices.shape[0]):
        r[i] = np.mean((dlogS[i, :] / dt))
    cov = np.cov(dlogS / np.sqrt(dt), bias=True)

    return r, cov
