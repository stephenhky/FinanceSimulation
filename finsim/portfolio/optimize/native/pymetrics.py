
import numpy as np
import numba as nb
from nptyping import NDArray, Shape, Float


@nb.njit
def python_sharpe_ratio(
        weights: NDArray[Shape["*"], Float],
        r: NDArray[Shape["*"], Float],
        cov: NDArray[Shape["*, *"], Float],
        rf: float
) -> float:
    yieldrate = np.sum(weights * r)
    sqweights = np.expand_dims(weights, axis=1) @ np.expand_dims(weights, axis=0)
    volatility = np.sqrt(np.sum(sqweights * cov))
    return (yieldrate - rf) / volatility


@nb.njit
def python_mpt_costfunction(
        weights: NDArray[Shape["*"], Float],
        r: NDArray[Shape["*"], Float],
        cov: NDArray[Shape["*, *"], Float],
        rf: float,
        lamb: float,
        V0: float=10.
) -> float:
    weightmat = np.expand_dims(weights[:-1], axis=0)
    c = lamb * V0
    return weights[-1]*rf + np.dot(weights[:-1], r) - 0.5*c/V0*(weightmat @ cov @ weightmat.T)[0, 0]


@nb.njit
def python_mpt_entropy_costfunction(
        weights: NDArray[Shape["*"], Float],
        r: NDArray[Shape["*"], Float],
        cov: NDArray[Shape["*, *"], Float],
        rf: float,
        lamb0: float,
        lamb1: float,
        V: float=10.
) -> float:
    weightmat = np.expand_dims(weights[:-1], axis=0)
    c0 = lamb0 * V
    c1 = lamb1 * V
    yield_val = weights[-1]*rf + np.dot(weights[:-1], r)
    cov_val = - 0.5 * c0 / V * (weightmat @ cov @ weightmat.T)[0, 0]
    sumweights = np.sum(weights[:-1])
    entropy_val = - 0.5 * c1 / V * np.sum(weights[:-1] * (np.log(weights[:-1]) - np.log(sumweights))) / sumweights
    return yield_val + cov_val + entropy_val
