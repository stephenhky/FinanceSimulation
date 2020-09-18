
from math import log

import numpy as np
import numba as nb


@nb.njit(nb.float64[:, :](nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.int64))
def simulate_BlackScholesMerton_stocks(S0, r, sigma, T, dt, nbsimulations):
    nbtimesteps = int(T // dt) + 1
    z = np.random.normal(size=(nbsimulations, nbtimesteps))
    logS = np.zeros((nbsimulations, nbtimesteps))
    logS[:, 0] = log(S0)
    for i in range(1, nbtimesteps):
        logS[:, i] = logS[:, i - 1] + \
                     (r - 0.5 * sigma * sigma) * dt + \
                     sigma * z[:, i] * np.sqrt(dt)
    return np.exp(logS)


