
import numpy as np


def estimate_downside_risk(ts, prices, target_return):
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    rarray = dlogS / dt
    less_return = rarray[rarray < target_return]
    if len(less_return) > 0:
        downside_risk = np.sqrt(np.mean(np.square(target_return - less_return)))
    else:
        downside_risk = 0.0
    return downside_risk
