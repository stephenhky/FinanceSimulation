
from .native.numbarisk import numba_estimate_downside_risk


def estimate_downside_risk(ts, prices, target_return):
    return numba_estimate_downside_risk(ts, prices, target_return)
