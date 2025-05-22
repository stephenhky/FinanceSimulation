
from typing import Literal

import numpy as np
from scipy import stats
from nptyping import NDArray, Shape, Float, Datetime64

from .native.pyrisk import python_estimate_downside_risk, python_estimate_upside_risk
from .constants import dividing_factors_dict


def estimate_downside_risk(
        timestamps: NDArray[Shape["*"], Datetime64],
        prices: NDArray[Shape["*"], Float],
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
        timestamps: NDArray[Shape["*"], Datetime64],
        prices: NDArray[Shape["*"], Float],
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
        timestamps: NDArray[Shape["*"], Datetime64],
        prices: NDArray[Shape["*"], Float],
        market_prices: NDArray[Shape["*"], Float],
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
