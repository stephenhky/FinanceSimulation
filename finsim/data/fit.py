
import numpy as np

from .native.numbafit import numba_fit_BlackScholesMerton_model, numba_fit_multivariate_BlackScholesMerton_model


dividing_factors_dict = {
    'second': 1.0,
    'minute': 60.0,
    'hour': 60.0*60.0,
    'day': 60.0*60.0*24.0,
    'year': 60.0*60.0*24.0*365.0
}


def fit_BlackScholesMerton_model(timestamps, prices, unit='year'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    # return numba_fit_BlackScholesMerton_model(ts, prices, dividing_factor)
    dlogS = np.log(prices[1:] / prices[:-1])
    dt = np.array(ts[1:] - ts[:-1], dtype='timedelta64[s]')
    dt = np.array(dt, dtype=np.float) / dividing_factor

    r = np.mean(dlogS / dt)
    sigma = np.std(dlogS / np.sqrt(dt))

    return r, sigma


def fit_multivariate_BlackScholesMerton_model(timestamps, multiprices, unit='year'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    # return numba_fit_multivariate_BlackScholesMerton_model(ts, multiprices, dividing_factor)
    dlogS = np.log(multiprices[:, 1:] / multiprices[:, :-1])
    dt = np.array(ts[1:] - ts[:-1], dtype='timedelta64[s]')
    dt = np.array(dt, dtype=np.float) / dividing_factor

    r = np.mean(dlogS / dt, axis=1)
    cov = np.cov(dlogS / np.sqrt(dt))

    return r, cov
