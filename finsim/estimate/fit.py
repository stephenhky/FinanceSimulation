
import numpy as np

from .constants import dividing_factors_dict
from .native.pyfit import python_fit_BlackScholesMerton_model, python_fit_multivariate_BlackScholesMerton_model
from .native.cythonfit import cython_fit_BlackScholesMerton_model, cython_fit_multivariate_BlackScholesMerton_model
from .native.fortranfit import fortranfit


# Note: always round-off to seconds first, but flexible about the unit to be used.

def fit_BlackScholesMerton_model(timestamps, prices, unit='year', lowlevellang='F'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    if lowlevellang == 'C':
        return cython_fit_BlackScholesMerton_model(ts, prices)
    elif lowlevellang == 'P':
        return python_fit_BlackScholesMerton_model(ts, prices)
    elif lowlevellang == 'F':
        results = fortranfit.f90_fit_blackscholesmerton_model(ts, prices)
        return results[0], results[1]
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "P" (Python), "C" (Cython), or "F" (Fortran).)'.format(
                lowlevellang))


def fit_multivariate_BlackScholesMerton_model(timestamps, multiprices, unit='year', lowlevellang='C'):
    dividing_factor = dividing_factors_dict[unit]

    ts = np.array(timestamps, dtype='datetime64[s]')
    ts = np.array(ts, dtype=np.float64) / dividing_factor

    if lowlevellang == 'C':
        return cython_fit_multivariate_BlackScholesMerton_model(ts, multiprices)
    elif lowlevellang == 'N':
        return python_fit_multivariate_BlackScholesMerton_model(ts, multiprices)
    else:
        raise ValueError(
            'Unknown low-level language: {}. (Should be "C" (Cython), or "F" (Fortran).)'.format(
                lowlevellang))
