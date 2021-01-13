
from functools import partial

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .metrics import sharpe_ratio, mpt_costfunction, mpt_entropy_costfunction


def optimized_portfolio_on_sharperatio(r, cov, rf, minweight=0.):
    func = partial(sharpe_ratio, r=r, cov=cov, rf=rf)
    nbstocks = len(r)
    initialguess = np.repeat(1 / nbstocks, nbstocks)
    constraints = [
        LinearConstraint(np.eye(nbstocks), minweight, 1.),
        LinearConstraint(np.array([np.repeat(1, nbstocks)]), 1, 1)
    ]
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )


def optimized_portfolio_mpt_costfunction(r, cov, rf, lamb, V0=10.):
    func = partial(mpt_costfunction, r=r, cov=cov, rf=rf, lamb=lamb, V0=V0)
    nbstocks = len(r)
    constraints = [
        LinearConstraint(np.eye(nbstocks+1), 0, V0)
    ]
    initialguess = np.repeat(V0 / (nbstocks+1), nbstocks+1)
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )


def optimized_portfolio_mpt_entropy_costfunction(r, cov, rf, lamb0, lamb1, V=10.):
    func = partial(mpt_entropy_costfunction, r=r, cov=cov, rf=rf, lamb0=lamb0, lamb1=lamb1, V=V)
    nbstocks = len(r)
    constraints = [
        LinearConstraint(np.eye(nbstocks+1), 0, V)
    ]
    initialguess = np.repeat(V / (nbstocks + 1), nbstocks + 1)
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )
