
import json
import logging
from collections import defaultdict

from tqdm import tqdm
import pandas as pd

from ..data.preader import get_yahoofinance_data, get_symbol_closing_price


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

        df = pd.DataFrame(stocks_data_dfs[max_timearray_ref]['TimeStamp'].copy())
        df['value'] = sum([
            self.symbols_nbshares[sym] * stocks_data_dfs[i]['Close']
            for i, sym in enumerate(self.symbols_nbshares.keys())
        ])
        return df

    def __add__(self, other):
        assert isinstance(other, Portfolio)

        symshares1 = defaultdict(lambda: 0, self.symbols_nbshares)
        symshares2 = defaultdict(lambda: 0, other.symbols_nbshares)
        all_symbols = set(symshares1.keys()).union(symshares2.keys())

        total_symbols_shares = {symbol: symshares2[symbol] + symshares1[symbol] for symbol in all_symbols}
        return Portfolio(total_symbols_shares, cacheddir=self.cacheddir)

    def __sub__(self, other):
        assert isinstance(other, Portfolio)

        symshares1 = defaultdict(lambda: 0, self.symbols_nbshares)
        symshares2 = defaultdict(lambda: 0, other.symbols_nbshares)
        all_symbols = set(symshares1.keys()).union(symshares2.keys())

        symbols_diff_shares = {symbol: symshares1[symbol] - symshares2[symbol] for symbol in all_symbols}
        return Portfolio(symbols_diff_shares, cacheddir=self.cacheddir)

    def __eq__(self, other):
        assert isinstance(other, Portfolio)

        for symbol, nbshares in self.symbols_nbshares.items():
            if symbol not in other.symbols_nbshares:
                return False
            if other.symbols_nbshares[symbol] != nbshares:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

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
            
    def __mul__(self, other):
        assert isinstance(other, int) or isinstance(other, float)
        newshares = {symbol: nbshares*other for symbol, nbshares in self.symbols_nbshares.items()}
        return Portfolio(newshares, cacheddir=self.cacheddir)

    def save_to_json(self, fileobj):
        json.dump(self.symbols_nbshares, fileobj)

    def dumps_json(self):
        return json.dumps(self.symbols_nbshares)

    @classmethod
    def load_from_json(cls, fileobj, cacheddir=None):
        symbols_nbshares = json.load(fileobj)
        return cls(symbols_nbshares, cacheddir=cacheddir)

    @classmethod
    def load_from_dict(cls, portdict, cacheddir=None):
        return cls(portdict, cacheddir=cacheddir)


class OptimizedPortfolio(Portfolio):
    def __init__(self, policy, totalworth, presetdate, cacheddir=None):
        super(OptimizedPortfolio, self).__init__({}, cacheddir=cacheddir)
        self.policy = policy
        self.totalworth = totalworth
        self.presetdate = presetdate
        self.compute()

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
        return Portfolio(self.symbols_nbshares, cacheddir=self.cacheddir)

    def save_to_json(self, fileobj):
        self.get_portfolio().save_to_json(fileobj)

    def dumps_json(self):
        return self.get_portfolio().dumps_json()

    @classmethod
    def load_from_json(cls, fileobj, cacheddir=None):
        raise NotImplementedError('OptimizedPortfolio does not implement loading from json. Use Portfolio to load instead.')

    @classmethod
    def load_from_dict(cls, portdict, cacheddir=None):
        raise NotImplementedError('OptimizedPortfolio does not implement loading from json. Use Portfolio to load instead.')
