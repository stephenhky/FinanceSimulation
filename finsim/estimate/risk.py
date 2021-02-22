
import numpy as np
from scipy import stats

from .native.numbarisk import numba_estimate_downside_risk, numba_estimate_upside_risk
from .constants import dividing_factors_dict


def estimate_downside_risk(timestamps, prices, target_return, unit='year'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float) / dividing_factor

    return numba_estimate_downside_risk(ts, prices, target_return)


def estimate_upside_risk(timestamps, prices, target_return, unit='year'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float) / dividing_factor

    return numba_estimate_upside_risk(ts, prices, target_return)


def estimate_beta(timestamps, prices, market_prices, unit='year'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float) / dividing_factor
    dt = ts[1:] - ts[:-1]

    dlogprices = np.log(prices[1:] / prices[:-1])
    dlogmarketprices = np.log(market_prices[1:] / market_prices[:-1])
    assert len(dt) == len(dlogprices)
    assert len(dt) == len(dlogmarketprices)

    stockyields = dlogprices / dt
    marketyields = dlogmarketprices / dt

    reg = stats.linregress(x=marketyields, y=stockyields)
    return reg.slope
