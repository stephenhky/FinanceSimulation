
from functools import partial
from itertools import product

import numpy as np
from scipy.optimize import minimize, LinearConstraint
import pandas as pd

from .metrics import sharpe_ratio
from ..data.preader import get_yahoofinance_data
from ..estimate.fit import fit_multivariate_BlackScholesMerton_model


def get_BlackScholesMerton_stocks_estimation(symbols, startdate, enddate):
    stocks_data_dfs = [
        get_yahoofinance_data(sym, startdate, enddate)
        for sym in symbols
    ]
    return fit_multivariate_BlackScholesMerton_model(
        np.array(stocks_data_dfs[0]['TimeStamp']),
        np.array([
            np.array(stocks_data_dfs[i]['Close'])
            for i in range(4)
        ])
    )


def optimized_portfolio_on_sharperatio(r, cov, rf):
    func = partial(sharpe_ratio, r=r, cov=cov, rf=rf)
    nbstocks = len(r)
    initialguess = np.repeat(1 /nbstocks, nbstocks)
    constraints = [
        LinearConstraint(np.eye(nbstocks), 0, 1),
        LinearConstraint(np.array([np.repeat(1, nbstocks)]), 1, 1)
    ]
    return minimize(
        lambda weights: -func(weights),
        initialguess,
        constraints=constraints
    )


class OptimizedPortfolio:
    def __init__(self, rf, r=None, cov=None, symbols=None):
        self.rf = rf
        self.optimized = False
        
        if r is not None and cov is not None:
            self.optimize(r, cov, symbols=symbols)

    def optimize(self, r, cov, symbols=None):
        assert len(r) == cov.shape[0]
        assert cov.shape[0] == cov.shape[1]
        if symbols is not None:
            assert len(r) == len(symbols)

        self.r = r
        self.cov = cov
        self.symbols = symbols if symbols is not None else list(range(len(r)))
        self.optimized_sol = optimized_portfolio_on_sharperatio(r, cov, self.rf)
        self.optimized = True

        self.optimized_weights = self.optimized_sol.x
        self.optimized_sharpe_ratio = -self.optimized_sol.fun
        self.optimized_portfolio_yield = np.sum(self.optimized_weights * self.r)
        sqweights = np.matmul(
            np.expand_dims(self.optimized_weights, axis=1),
            np.expand_dims(self.optimized_weights, axis=0)
        )
        self.optimized_volatility = np.sqrt(np.sum(sqweights * self.cov))

    @property
    def weights(self):
        return self.optimized_weights

    @property
    def portfolio_yield(self):
        return self.optimized_portfolio_yield

    @property
    def volatility(self):
        return self.optimized_volatility

    @property
    def sharpe_ratio(self):
        return self.optimized_sharpe_ratio

    @property
    def correlation_matrix(self):
        corr = np.zeros(self.cov.shape)
        for i, j in product(range(self.cov.shape[0]), range(self.cov.shape[1])):
            corr[i, j] = self.cov[i, j] / np.sqrt(self.cov[i, i] * self.cov[j, j])
        return corr

    @property
    def named_correlation_matrix(self):
        corrdf = pd.DataFrame(self.correlation_matrix,
                              columns=self.symbols,
                              index=self.symbols)
        return corrdf

    @property
    def portfolio_summary(self):
        summary = {
            'components': [
                {
                    'symbol': sym,
                    'yield': self.r[i],
                    'weight': self.weights[i],
                    'volatility': self.cov[i, i]
                }
                for i, sym in enumerate(self.symbols)
            ],
            'yield': self.optimized_portfolio_yield,
            'volatility': self.optimized_volatility,
            'sharpe_ratio': self.optimized_sharpe_ratio,
            'correlation': self.correlation_matrix
        }
        return summary