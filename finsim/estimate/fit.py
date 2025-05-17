
from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt
from itertools import product

from .constants import dividing_factors_dict
from .native.pyfit import python_fit_BlackScholesMerton_model, python_fit_multivariate_BlackScholesMerton_model


# Note: always round-off to seconds first, but flexible about the unit to be used.

def fit_BlackScholesMerton_model(
        timestamps: npt.NDArray[np.datetime64],
        prices: npt.NDArray[np.float64],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year',
        lowlevellang: Literal['C', 'P']='P'
) -> Tuple[float, float]:
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'P':
        return python_fit_BlackScholesMerton_model(ts, prices)
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "P" (Python), or "C" (Cython).)'.format(
                lowlevellang))


def fit_multivariate_BlackScholesMerton_model(
        timestamps: npt.NDArray[np.datetime64],
        multiprices: npt.NDArray[np.float64],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year',
        lowlevellang: Literal['C', 'P']='P'
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    if lowlevellang == 'C':
        raise ValueError("Cython fitting is no longer supported!")
    elif lowlevellang == 'P':
        return python_fit_multivariate_BlackScholesMerton_model(ts, multiprices)
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "C" (Cython).)'.format(
                lowlevellang))


######## routines below are for time-weighted portfolio building

def fit_timeweighted_BlackScholesMerton_model(
        timestamps: npt.NDArray[np.datetime64],
        prices: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year'
) -> Tuple[float, float]:
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    dlogS = np.log(prices[1:] / prices[:-1])
    dt = ts[1:] - ts[:-1]

    r = np.average(dlogS / dt, weights=weights[1:])
    sigma = np.sqrt(np.average(np.square(dlogS / np.sqrt(dt)), weights=weights[1:]) - np.square(
        np.average(dlogS / np.sqrt(dt), weights=weights[1:])))

    return r, sigma


def fit_timeweighted_multivariate_BlackScholesMerton_model(
        timestamps: npt.NDArray[np.datetime64],
        multiprices: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        unit: Literal['second', 'minute', 'hour', 'day', 'year']='year'
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    # possibly hit this part
    dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    dt = ts[1:] - ts[:-1]

    # estimation
    r = np.zeros(multiprices.shape[0])
    for i in range(multiprices.shape[0]):
        r[i] = np.average((dlogS[i, :] / dt), weights=weights[1:])
    cov = np.zeros((multiprices.shape[0], multiprices.shape[0]))
    for i, j in product(range(multiprices.shape[0]), range(multiprices.shape[0])):
        avg_i = np.average(dlogS[i, :] / np.sqrt(dt), weights=weights[1:])
        avg_j = np.average(dlogS[j, :] / np.sqrt(dt), weights=weights[1:])
        cov[i, j] = np.average(dlogS[i, :] * dlogS[j, :] / dt, weights=np.square(weights[1:])) - avg_i * avg_j

    return r, cov
