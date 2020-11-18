
import numpy as np
import numba as nb


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64))
def numba_estimate_downside_risk(ts, prices, target_return):
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    rarray = dlogS / dt
    less_return_array = target_return - rarray
    less_return_array[less_return_array < 0] = 0.
    downside_risk = np.sqrt(np.mean(np.square(less_return_array)))
    return downside_risk
