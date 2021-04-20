
import numpy as np


def python_estimate_downside_risk(ts, prices, target_return):
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    rms_rarray = dlogS / np.sqrt(dt)
    less_return_array = target_return - rms_rarray
    less_return_array[less_return_array < 0] = 0.
    downside_risk = np.sqrt(np.mean(np.square(less_return_array)))
    return downside_risk


def python_estimate_upside_risk(ts, prices, target_return):
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    rms_rarray = dlogS / np.sqrt(dt)
    more_return_array = rms_rarray - target_return
    more_return_array[more_return_array < 0] = 0.
    upside_risk = np.sqrt(np.mean(np.square(more_return_array)))
    return upside_risk
