
import numpy as np

from .constants import dividing_factors_dict
from .native.numbafit import numba_fit_BlackScholesMerton_model, numba_fit_multivariate_BlackScholesMerton_model


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
