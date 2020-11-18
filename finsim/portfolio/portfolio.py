
import logging

from tqdm import tqdm
import pandas as pd

from .optimize.policy import OptimizedWeightingPolicyUsingMPTSharpeRatio
from .numerics import get_BlackScholesMerton_stocks_estimation
from .numerics import get_symbol_closing_price
from ..data.preader import get_yahoofinance_data


class Portfolio:
    def __init__(self, symbols_nbshares, cacheddir=None):   # symbols_nbshares = {'NVDA': 200, 'AMZN': 101}
        self.symbols_nbshares = symbols_nbshares
        self.cacheddir = cacheddir

    def get_portfolio_value(self, datestr):
        portfolio_value = sum([
            self.symbols_nbshares[symbol] * get_symbol_closing_price(symbol, datestr, cacheddir=self.cacheddir)
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


class OptimizedPortfolio(Portfolio):
    def __init__(self, policy, totalworth, presetdate, cacheddir=None):
        super(OptimizedPortfolio, self).__init__({})
        self.policy = policy
        self.totalworth = totalworth
        self.presetdate = presetdate
        self.compute()
        self.cacheddir = cacheddir

    def compute(self):
        prices = {
            symbol: get_symbol_closing_price(symbol, self.presetdate, cacheddir=self.cacheddir)
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
    optimized_weighting_policy = OptimizedWeightingPolicyUsingMPTSharpeRatio(rf, r, cov, symbols, minweight=minweight)
    optimized_portfolio = OptimizedPortfolio(optimized_weighting_policy, totalworth, presetdate)
    return optimized_portfolio
