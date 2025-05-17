
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

from .native.pyrisk import python_estimate_downside_risk, python_estimate_upside_risk
from .constants import dividing_factors_dict


def estimate_downside_risk(
        timestamps: npt.NDArray[np.datetime64],
        prices: npt.NDArray[np.float64],
        target_return: float,
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year',
        lowlevellang: Literal['C', 'P']='P'
) -> float:
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'P':
        return python_estimate_downside_risk(ts, prices, target_return)
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython))'.format(
                lowlevellang))


def estimate_upside_risk(
        timestamps: npt.NDArray[np.datetime64],
        prices: npt.NDArray[np.float64],
        target_return: float,
        unit: Literal['second', 'minute', 'hour', 'day', 'year'] = 'year',
        lowlevellang: Literal['C', 'P'] = 'P'
) -> float:
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'N':
        return python_estimate_upside_risk(ts, prices, target_return)
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(
                lowlevellang))


def estimate_beta(
        timestamps: npt.NDArray[np.datetime64],
        prices: npt.NDArray[np.float64],
        market_prices: npt.NDArray[np.float64],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year'
) -> float:
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor
    dt = ts[1:] - ts[:-1]

    dlogprices = np.log(prices[1:] / prices[:-1])
    dlogmarketprices = np.log(market_prices[1:] / market_prices[:-1])
    assert len(dt) == len(dlogprices)
    assert len(dt) == len(dlogmarketprices)

    stockyields = dlogprices / dt
    marketyields = dlogmarketprices / dt

    reg = stats.linregress(x=marketyields, y=stockyields)
    return reg.slope
