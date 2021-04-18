
import numpy as np
from scipy import stats

from .native.numbarisk import numba_estimate_downside_risk, numba_estimate_upside_risk
from .native.cythonrisk import cython_estimate_downside_risk, cython_estimate_upside_risk
from .constants import dividing_factors_dict


def estimate_downside_risk(timestamps, prices, target_return, unit='year', lowlevellang='C'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float) / dividing_factor

    if lowlevellang == 'C':
        return cython_estimate_downside_risk(ts, prices, target_return)
    elif lowlevellang == 'N':
        return numba_estimate_downside_risk(ts, prices, target_return)
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "N" (numba), "C" (Cython), or "F" (Fortran).)'.format(
                lowlevellang))


def estimate_upside_risk(timestamps, prices, target_return, unit='year', lowlevellang='C'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float) / dividing_factor

    if lowlevellang == 'C':
        return cython_estimate_upside_risk(ts, prices, target_return)
    elif lowlevellang == 'N':
        return numba_estimate_upside_risk(ts, prices, target_return)
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "N" (numba), "C" (Cython), or "F" (Fortran).)'.format(
                lowlevellang))


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
