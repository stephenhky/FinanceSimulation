
import numpy as np


def sharpe_ratio(weights, r, cov, rf):
    yieldrate = np.sum(weights * r)
    sqweights = np.matmul(
        np.expand_dims(weights, axis=1),
        np.expand_dims(weights, axis=0)
    )
    volatility = np.sqrt(np.sum(sqweights * cov))
    return (yieldrate - rf) / volatility
