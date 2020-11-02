
import sys
from itertools import product
import logging

from tqdm import tqdm
import numpy as np
import pandas as pd

from .numerics import get_BlackScholesMerton_stocks_estimation
from .numerics import get_symbol_closing_price, optimized_portfolio_on_sharperatio
from ..data.preader import get_yahoofinance_data


class Portfolio:
    def __init__(self, symbols_nbshares):   # symbols_nbshares = {'NVDA': 200, 'AMZN': 101}
        self.symbols_nbshares = symbols_nbshares

    def get_portfolio_value(self, datestr):
        portfolio_value = sum([
            self.symbols_nbshares[symbol] * get_symbol_closing_price(symbol, datestr)
            for symbol in self.symbols_nbshares
        ])
        return portfolio_value

    def get_portfolio_values_overtime(self, startdate, enddate, cacheddir=None):
        logging.info('Reading financial data...')
        stocks_data_dfs = [
            get_yahoofinance_data(sym, startdate, enddate, cacheddir=cacheddir)
            for sym in tqdm(self.symbols_nbshares.keys())
        ]

        logging.info('Estimating...')
        max_timearray_ref = 0
        maxlen = max(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)))
        minlen = min(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)))
        if minlen != maxlen:
            logging.warning('Not all symbols have data all the way back to {}'.format(startdate))
            max_timearray_ref = [i for i in range(len(stocks_data_dfs)) if maxlen == len(stocks_data_dfs[i])][0]
            logging.warning('Symbols not having whole range of data:')
            for i, symbol in enumerate(self.symbols_nbshares):
                thisdflen = len(stocks_data_dfs[i])
                if thisdflen != maxlen:
                    if thisdflen == 0:
                        logging.warning('No data for {} for this date range at all.'.format(symbol))
                        predf = pd.DataFrame(stocks_data_dfs[max_timearray_ref]['TimeStamp'])
                    else:
                        logging.warning('{}: starting from {}'.format(symbol, stocks_data_dfs[i]['TimeStamp'][0].date().strftime('%Y-%m-%d')))
                        predf = pd.DataFrame(stocks_data_dfs[max_timearray_ref]['TimeStamp'][:(maxlen - thisdflen)])
                    predf[stocks_data_dfs[max_timearray_ref].columns[1:]] = 0
                    stocks_data_dfs[i] = predf.append(stocks_data_dfs[i])

        df = pd.DataFrame(stocks_data_dfs[max_timearray_ref]['TimeStamp'])
        df['value'] = sum([
            self.symbols_nbshares[sym] * stocks_data_dfs[i]['Close']
            for i, sym in enumerate(self.symbols_nbshares.keys())
        ])
        return df

    @property
    def portfolio_symbols_nbshares(self):
        return self.symbols_nbshares

    def roundoff_nbshares(self):
        for symbol in self.symbols_nbshares:
            nbshares = round(self.symbols_nbshares[symbol])
            self.symbols_nbshares[symbol] = nbshares

    def multiply(self, factor):
        for symbol in self.symbols_nbshares:
            nbshares = self.symbols_nbshares[symbol]
            self.symbols_nbshares[symbol] = nbshares * factor


class OptimizedWeightingPolicy:
    def __init__(self, rf, r=None, cov=None, symbols=None, minweight=0.):
        self.rf = rf
        self.optimized = False
        self.minweight = minweight
        
        if r is not None and cov is not None:
            self.optimize(r, cov, symbols=symbols)

    def optimize_modern_portfolio_theory(self, r, cov, symbols=None):
        assert len(r) == cov.shape[0]
        assert cov.shape[0] == cov.shape[1]
        if symbols is not None:
            assert len(r) == len(symbols)

        self.r = r
        self.cov = cov
        self.symbols = symbols if symbols is not None else list(range(len(r)))
        self.optimized_sol = optimized_portfolio_on_sharperatio(r, cov, self.rf, minweight=self.minweight)
        self.optimized = True

        self.optimized_weights = self.optimized_sol.x
        self.optimized_sharpe_ratio = -self.optimized_sol.fun
        self.optimized_portfolio_yield = np.sum(self.optimized_weights * self.r)
        sqweights = np.matmul(
            np.expand_dims(self.optimized_weights, axis=1),
            np.expand_dims(self.optimized_weights, axis=0)
        )
        self.optimized_volatility = np.sqrt(np.sum(sqweights * self.cov))

    def optimize(self, r, cov, symbols=None, theory='MPT'):
        if theory.upper() == 'MPT':
            return self.optimize_modern_portfolio_theory(r, cov, symbols=symbols)
        else:
            pass

    @property
    def portfolio_symbols(self):
        return self.symbols

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
                    'volatility': np.sqrt(self.cov[i, i])
                }
                for i, sym in enumerate(self.symbols)
            ],
            'yield': self.optimized_portfolio_yield,
            'volatility': self.optimized_volatility,
            'sharpe_ratio': self.optimized_sharpe_ratio,
            'correlation': self.correlation_matrix
        }
        return summary


class OptimizedPortfolio(Portfolio):
    def __init__(self, policy, totalworth, presetdate):
        super(OptimizedPortfolio, self).__init__({})
        self.policy = policy
        self.totalworth = totalworth
        self.presetdate = presetdate
        self.compute()

    def compute(self):
        prices = {
            symbol: get_symbol_closing_price(symbol, self.presetdate)
            for symbol in self.policy.symbols
        }
        summary = self.policy.portfolio_summary
        for component in summary['components']:
            symbol = component['symbol']
            component['nbshares'] = component['weight'] * self.totalworth / prices[symbol]
            self.symbols_nbshares[symbol] = component['nbshares']

        self.summary = summary

    @property
    def portfolio_symbols(self):
        return self.policy.portfolio_symbols

    @property
    def weights(self):
        return self.policy.weights

    @property
    def portfolio_yield(self):
        return self.policy.portfolio_yield

    @property
    def volatility(self):
        return self.policy.volatility

    @property
    def sharpe_ratio(self):
        return self.policy.sharpe_ratio

    @property
    def correlation_matrix(self):
        return self.policy.correlation_matrix

    @property
    def named_correlation_matrix(self):
        return self.policy.named_correlation_matrix

    @property
    def portfolio_summary(self):
        return self.summary

    def get_portfolio(self):
        return Portfolio(self.symbols_nbshares)


def get_optimized_portfolio(
        rf,
        symbols,
        totalworth,
        presetdate,
        estimating_startdate,
        estimating_enddate,
        minweight=0.,
        lazy=False,
        cacheddir=None
):
    r, cov = get_BlackScholesMerton_stocks_estimation(
        symbols,
        estimating_startdate,
        estimating_enddate,
        lazy=lazy,
        cacheddir=cacheddir
    )
    optimized_weighting_policy = OptimizedWeightingPolicy(rf, r, cov, symbols, minweight=minweight)
    optimized_portfolio = OptimizedPortfolio(optimized_weighting_policy, totalworth, presetdate)
    return optimized_portfolio
