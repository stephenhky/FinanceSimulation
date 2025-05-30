
import numpy as np
import numba as nb
from nptyping import NDArray, Shape, Float


@nb.njit
def python_estimate_downside_risk(
        ts: NDArray[Shape["*"], Float],
        prices: NDArray[Shape["*"], Float],
        target_return: float
) -> float:
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    rms_rarray = dlogS / np.sqrt(dt)
    less_return_array = target_return - rms_rarray
    less_return_array[less_return_array < 0] = 0.
    downside_risk = np.sqrt(np.mean(np.square(less_return_array)))
    return downside_risk


@nb.njit
def python_estimate_upside_risk(
        ts: NDArray[Shape["*"], Float],
        prices: NDArray[Shape["*"], Float],
        target_return: float
) -> float:
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    rms_rarray = dlogS / np.sqrt(dt)
    more_return_array = rms_rarray - target_return
    more_return_array[more_return_array < 0] = 0.
    upside_risk = np.sqrt(np.mean(np.square(more_return_array)))
    return upside_risk
