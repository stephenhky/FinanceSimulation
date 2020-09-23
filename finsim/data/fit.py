
import numpy as np

from .native.numbafit import numba_fit_BlackScholesMerton_model, numba_fit_multivariate_BlackScholesMerton_model


dividing_factors_dict = {
    'second': 1.0,
    'minute': 60.0,
    'hour': 60.0*60.0,
    'day': 60.0*60.0*24.0,
    'year': 60.0*60.0*24.0*365.0
}


# Note: always round-off to seconds first, but flexible about the unit to be used.

def fit_BlackScholesMerton_model(timestamps, prices, unit='year'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float) / dividing_factor
    return numba_fit_BlackScholesMerton_model(ts, prices)


def fit_multivariate_BlackScholesMerton_model(timestamps, multiprices, unit='year'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float) / dividing_factor
    return numba_fit_multivariate_BlackScholesMerton_model(ts, multiprices)
