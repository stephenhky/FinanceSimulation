
import json
import logging
from collections import defaultdict
from typing import Union, Self, Any
from pathlib import Path
from io import TextIOWrapper

from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..data.preader import get_yahoofinance_data, get_symbol_closing_price
from .optimize.policy import OptimizedWeightingPolicy
from .helper import align_timestamps_stock_dataframes


class Portfolio:
    def __init__(
            self,
            symbols_nbshares: dict[str, Union[int, float]],    # e.g., symbols_nbshares = {'NVDA': 200, 'AMZN': 101}
            cacheddir: Union[Path, str]=None
    ):
        self.symbols_nbshares = symbols_nbshares
        self.cacheddir = cacheddir

    def get_portfolio_value(self, datestr: str) -> float:
        portfolio_value = sum([
            self.symbols_nbshares[symbol] * get_symbol_closing_price(symbol, datestr, cacheddir=self.cacheddir)
            for symbol in self.symbols_nbshares
        ])
        return portfolio_value

    def get_portfolio_values_overtime(
            self,
            startdate: str,
            enddate: str,
            cacheddir: Union[Path, str]=None,
            progressbar: bool=False
    ) -> pd.DataFrame:
        logging.debug('Reading financial data...')
        iterator = tqdm(self.symbols_nbshares.keys()) if progressbar else self.symbols_nbshares.keys()
        stocks_data_dfs = [
            get_yahoofinance_data(sym, startdate, enddate, cacheddir=cacheddir)
            for sym in iterator
        ]

        # unify the timestamps columns
        logging.info('Unifying timestamps....')
        stocks_data_dfs = align_timestamps_stock_dataframes(stocks_data_dfs)

        logging.debug('Estimating...')
        max_timearray_ref = 0
        maxlen = max(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)))
        minlen = min(len(stocks_data_dfs[i]) for i in range(len(stocks_data_dfs)))

        if minlen != maxlen:
            logging.warning(f'Not all symbols have data all the way back to {startdate}')
            max_timearray_ref = [i for i in range(len(stocks_data_dfs)) if maxlen == len(stocks_data_dfs[i])][0]
            logging.warning('Symbols not having whole range of data:')
            for i, symbol in enumerate(self.symbols_nbshares):
                thisdflen = len(stocks_data_dfs[i])
                if thisdflen != maxlen:
                    if thisdflen == 0:
                        logging.warning(f'No data for {symbol} for this date range at all.')
                        predf = pd.DataFrame(stocks_data_dfs[max_timearray_ref]['TimeStamp'])
                    else:
                        startdatestr = stocks_data_dfs[i].loc[0, 'TimeStamp'].date().strftime('%Y-%m-%d')
                        logging.warning(f'{symbol}: starting from {startdatestr}')
                        predf = pd.DataFrame(stocks_data_dfs[max_timearray_ref].loc[:(maxlen - thisdflen), 'TimeStamp'])
                    predf[stocks_data_dfs[max_timearray_ref].columns[1:]] = 0
                    predf = predf.append(stocks_data_dfs[i]).reset_index()
                    del predf['index']
                    stocks_data_dfs[i] = predf

        df = pd.DataFrame(stocks_data_dfs[max_timearray_ref]['TimeStamp'].copy())
        df['value'] = sum([
            self.symbols_nbshares[sym] * stocks_data_dfs[i]['Close']
            for i, sym in enumerate(self.symbols_nbshares.keys())
        ])
        return df

    def __add__(self, other: Self) -> Self:
        assert isinstance(other, Portfolio)

        symshares1 = defaultdict(lambda: 0, self.symbols_nbshares)
        symshares2 = defaultdict(lambda: 0, other.symbols_nbshares)
        all_symbols = set(symshares1.keys()).union(symshares2.keys())

        total_symbols_shares = {symbol: symshares2[symbol] + symshares1[symbol] for symbol in all_symbols}
        return Portfolio(total_symbols_shares, cacheddir=self.cacheddir)

    def __sub__(self, other: Self) -> Self:
        assert isinstance(other, Portfolio)

        symshares1 = defaultdict(lambda: 0, self.symbols_nbshares)
        symshares2 = defaultdict(lambda: 0, other.symbols_nbshares)
        all_symbols = set(symshares1.keys()).union(symshares2.keys())

        symbols_diff_shares = {symbol: symshares1[symbol] - symshares2[symbol] for symbol in all_symbols}
        return Portfolio(symbols_diff_shares, cacheddir=self.cacheddir)

    def __eq__(self, other: Self) -> bool:
        assert isinstance(other, Portfolio)

        for symbol, nbshares in self.symbols_nbshares.items():
            if symbol not in other.symbols_nbshares:
                return False
            if other.symbols_nbshares[symbol] != nbshares:
                return False

        return True

    def __ne__(self, other: Self) -> bool:
        return not self.__eq__(other)

    @property
    def portfolio_symbols_nbshares(self) -> dict[str, Union[int, float]]:
        return self.symbols_nbshares

    def roundoff_nbshares(self) -> None:
        for symbol in self.symbols_nbshares:
            nbshares = round(self.symbols_nbshares[symbol])
            self.symbols_nbshares[symbol] = nbshares

    def multiply(self, factor: float) -> None:
        for symbol in self.symbols_nbshares:
            nbshares = self.symbols_nbshares[symbol]
            self.symbols_nbshares[symbol] = nbshares * factor
            
    def __mul__(self, other: dict[str, Union[int, float]]) -> Self:
        assert isinstance(other, int) or isinstance(other, float)
        newshares = {symbol: nbshares*other for symbol, nbshares in self.symbols_nbshares.items()}
        return Portfolio(newshares, cacheddir=self.cacheddir)

    def save_to_json(self, fileobj: TextIOWrapper) -> None:
        json.dump(self.symbols_nbshares, fileobj)

    def dumps_json(self) -> str:
        return json.dumps(self.symbols_nbshares)

    @classmethod
    def load_from_json(cls, fileobj: TextIOWrapper, cacheddir: Union[Path, str]=None) -> Self:
        symbols_nbshares = json.load(fileobj)
        return cls(symbols_nbshares, cacheddir=cacheddir)

    @classmethod
    def load_from_dict(cls, portdict: dict[str, Union[int, float]], cacheddir: Union[Path, str]=None) -> Self:
        return cls(portdict, cacheddir=cacheddir)


class OptimizedPortfolio(Portfolio):
    def __init__(
            self,
            policy: OptimizedWeightingPolicy,
            totalworth: float,
            presetdate: str,
            cacheddir: Union[Path, str]=None
    ):
        super(OptimizedPortfolio, self).__init__({}, cacheddir=cacheddir)
        self.policy = policy
        self.totalworth = totalworth
        self.presetdate = presetdate
        self.compute()

    def compute(self) -> None:
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
    def portfolio_symbols(self) -> list[str]:
        return self.policy.portfolio_symbols

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        return self.policy.weights

    @property
    def portfolio_yield(self) -> float:
        return self.policy.portfolio_yield

    @property
    def volatility(self) -> float:
        return self.policy.volatility

    @property
    def correlation_matrix(self) -> npt.NDArray[np.float64]:
        return self.policy.correlation_matrix

    @property
    def named_correlation_matrix(self) -> pd.DataFrame:
        return self.policy.named_correlation_matrix

    @property
    def portfolio_summary(self) -> dict[str, Any]:
        return self.summary

    def get_portfolio(self) -> Portfolio:
        return Portfolio(self.symbols_nbshares, cacheddir=self.cacheddir)

    def save_to_json(self, fileobj: TextIOWrapper) -> None:
        self.get_portfolio().save_to_json(fileobj)

    def dumps_json(self) -> str:
        return self.get_portfolio().dumps_json()

    @classmethod
    def load_from_json(cls, fileobj: TextIOWrapper, cacheddir: Union[Path, str]=None) -> Self:
        raise NotImplementedError('OptimizedPortfolio does not implement loading from json. Use Portfolio to load instead.')

    @classmethod
    def load_from_dict(cls, portdict: TextIOWrapper, cacheddir: Union[Path, str]=None) -> Self:
        raise NotImplementedError('OptimizedPortfolio does not implement loading from json. Use Portfolio to load instead.')
