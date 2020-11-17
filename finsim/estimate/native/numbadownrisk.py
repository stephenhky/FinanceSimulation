
import numpy as np
import numba as nb


@nb.njit(nb.float64(nb.float64[:], nb.float64[:], nb.float64))
def numba_estimate_downside_risk(ts, prices, target_return):
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    rarray = dlogS / dt
    downside_risk = np.sqrt(np.mean(np.square(np.max(0, target_return-rarray))))
    return downside_risk
