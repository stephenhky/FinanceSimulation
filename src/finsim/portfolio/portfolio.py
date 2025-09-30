
import json
import logging
import sys
from collections import defaultdict
from typing import Any, Optional, Literal, Annotated
from os import PathLike
from io import TextIOWrapper
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import pandas as pd

from ..data.preader import get_yahoofinance_data, get_symbol_closing_price
from .optimize.policy import OptimizedWeightingPolicy
from .helper import align_timestamps_stock_dataframes
from ..schemas.portfolios import PortfolioSchema


class Portfolio:
    """A class representing a portfolio of financial assets.
    
    This class manages a collection of assets (stocks) with their respective quantities
    and provides methods for calculating portfolio values, performing operations on
    portfolios, and saving/loading portfolio data.
    """
    
    def __init__(
            self,
            symbols_nbshares: dict[str, int | float],    # e.g., symbols_nbshares = {'NVDA': 200, 'AMZN': 101}
            cacheddir: Optional[PathLike | str]=None
    ):
        """Initialize a Portfolio with asset symbols and quantities.
        
        Args:
            symbols_nbshares: Dictionary mapping stock symbols to number of shares
            cacheddir: Directory for cached data (optional)
        """
        validated_portdict = PortfolioSchema.model_validate({'content': symbols_nbshares}).model_dump().get('content')

        self.symbols_nbshares = validated_portdict
        self.cacheddir = cacheddir

    def get_portfolio_value(self, datestr: str) -> float:
        """Calculate the total value of the portfolio on a specific date.
        
        Args:
            datestr: Date in 'YYYY-MM-DD' format
            
        Returns:
            float: Total portfolio value on the specified date
        """
        portfolio_value = sum([
            self.symbols_nbshares[symbol] * get_symbol_closing_price(symbol, datestr, cacheddir=self.cacheddir)
            for symbol in self.symbols_nbshares
        ])
        return portfolio_value

    def get_portfolio_values_overtime(
            self,
            startdate: str,
            enddate: str,
            cacheddir: Optional[PathLike | str]=None,
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
        """Add two portfolios together.
        
        This method combines two portfolios by adding the quantities of the same assets.
        
        Args:
            other: Another Portfolio object to add to this one
            
        Returns:
            Self: A new Portfolio object representing the combined portfolios
        """
        assert isinstance(other, Portfolio)

        symshares1 = defaultdict(lambda: 0, self.symbols_nbshares)
        symshares2 = defaultdict(lambda: 0, other.symbols_nbshares)
        all_symbols = set(symshares1.keys()).union(symshares2.keys())

        total_symbols_shares = {symbol: symshares2[symbol] + symshares1[symbol] for symbol in all_symbols}
        return Portfolio(total_symbols_shares, cacheddir=self.cacheddir)

    def __sub__(self, other: Self) -> Self:
        """Subtract one portfolio from another.
        
        This method subtracts the quantities of assets in one portfolio from another.
        
        Args:
            other: Another Portfolio object to subtract from this one
            
        Returns:
            Self: A new Portfolio object representing the difference between portfolios
        """
        assert isinstance(other, Portfolio)

        symshares1 = defaultdict(lambda: 0, self.symbols_nbshares)
        symshares2 = defaultdict(lambda: 0, other.symbols_nbshares)
        all_symbols = set(symshares1.keys()).union(symshares2.keys())

        symbols_diff_shares = {
            symbol: symshares1[symbol] - symshares2[symbol]
            for symbol in all_symbols
        }
        symbols_diff_shares = {
            symbol: nbshares
            for symbol, nbshares in symbols_diff_shares.items()
            if nbshares != 0
        }
        return Portfolio(symbols_diff_shares, cacheddir=self.cacheddir)

    def __eq__(self, other: Self) -> bool:
        """Check if two portfolios are equal.
        
        This method compares two portfolios to see if they have the same assets
        and quantities.
        
        Args:
            other: Another Portfolio object to compare with this one
            
        Returns:
            bool: True if portfolios are equal, False otherwise
        """
        assert isinstance(other, Portfolio)

        for symbol, nbshares in self.symbols_nbshares.items():
            if symbol not in other.symbols_nbshares:
                return False
            if other.symbols_nbshares[symbol] != nbshares:
                return False

        return True

    def __ne__(self, other: Self) -> bool:
        """Check if two portfolios are not equal.
        
        This method compares two portfolios to see if they have different assets
        or quantities.
        
        Args:
            other: Another Portfolio object to compare with this one
            
        Returns:
            bool: True if portfolios are not equal, False otherwise
        """
        return not self.__eq__(other)

    @property
    def portfolio_symbols_nbshares(self) -> dict[str, int | float]:
        """Get the dictionary of symbols and number of shares.
        
        Returns:
            dict[str, Union[int, float]]: Dictionary mapping symbols to number of shares
        """
        return self.symbols_nbshares

    def roundoff_nbshares(self) -> None:
        """Round off the number of shares for all assets to the nearest integer.
        
        This method modifies the portfolio in-place by rounding the number of
        shares for each asset to the nearest integer.
        """
        for symbol in self.symbols_nbshares:
            nbshares = round(self.symbols_nbshares[symbol])
            self.symbols_nbshares[symbol] = nbshares

    def multiply(self, factor: float) -> None:
        """Multiply the number of shares for all assets by a factor.
        
        This method modifies the portfolio in-place by multiplying the number of
        shares for each asset by the specified factor.
        
        Args:
            factor: The factor by which to multiply the number of shares
        """
        for symbol in self.symbols_nbshares:
            nbshares = self.symbols_nbshares[symbol]
            self.symbols_nbshares[symbol] = nbshares * factor
            
    def __mul__(self, other: int | float) -> Self:
        """Multiply the portfolio by a scalar factor.
        
        This method creates a new portfolio with the number of shares for each
        asset multiplied by the specified factor.
        
        Args:
            other: The factor by which to multiply the number of shares
            
        Returns:
            Self: A new Portfolio object with multiplied share quantities
        """
        assert isinstance(other, int) or isinstance(other, float)
        newshares = {symbol: nbshares*other for symbol, nbshares in self.symbols_nbshares.items()}
        return Portfolio(newshares, cacheddir=self.cacheddir)

    def __rmul__(self, other: int | float) -> Self:
        """Multiply the portfolio by a scalar factor.

        This method creates a new portfolio with the number of shares for each
        asset multiplied by the specified factor.

        Args:
            other: The factor by which to multiply the number of shares

        Returns:
            Self: A new Portfolio object with multiplied share quantities
        """
        return self * other

    def save_to_json(self, fileobj: TextIOWrapper) -> None:
        """Save the portfolio to a JSON file.
        
        Args:
            fileobj: File object to write the portfolio data to
        """
        json.dump(self.symbols_nbshares, fileobj)

    def dumps_json(self) -> str:
        """Serialize the portfolio to a JSON string.
        
        Returns:
            str: JSON string representation of the portfolio
        """
        return json.dumps(self.symbols_nbshares)

    @classmethod
    def load_from_json(
            cls,
            fileobj: TextIOWrapper,
            cacheddir: Optional[PathLike | str]=None
    ) -> Self:
        """Load a portfolio from a JSON file.
        
        Args:
            fileobj: File object to read the portfolio data from
            cacheddir: Directory for cached data (optional)
            
        Returns:
            Self: A new Portfolio object loaded from the JSON file
        """
        symbols_nbshares = json.load(fileobj)
        return cls.load_from_dict(symbols_nbshares, cacheddir=cacheddir)

    @classmethod
    def load_from_dict(
            cls,
            portdict: dict[str, int | float],
            cacheddir: Optional[PathLike | str]=None
    ) -> Self:
        """Load a portfolio from a dictionary.
        
        Args:
            portdict: Dictionary mapping symbols to number of shares
            cacheddir: Directory for cached data (optional)
            
        Returns:
            Self: A new Portfolio object loaded from the dictionary
        """
        return cls(portdict, cacheddir=cacheddir)


class OptimizedPortfolio(Portfolio):
    """A class representing an optimized portfolio of financial assets.
    
    This class extends the basic Portfolio class to include optimization features
    based on Modern Portfolio Theory, with methods for calculating optimal weights
    and portfolio metrics.
    """
    
    def __init__(
            self,
            policy: OptimizedWeightingPolicy,
            totalworth: float,
            presetdate: str,
            cacheddir: Optional[PathLike | str]=None
    ):
        """Initialize an OptimizedPortfolio with an optimization policy.
        
        Args:
            policy: The optimization policy to use for calculating weights
            totalworth: Total value of the portfolio
            presetdate: Date for which to calculate the portfolio composition
            cacheddir: Directory for cached data (optional)
        """
        super(OptimizedPortfolio, self).__init__({}, cacheddir=cacheddir)
        self.policy = policy
        self.totalworth = totalworth
        self.presetdate = presetdate
        self.compute()

    def compute(self) -> None:
        """Compute the optimized portfolio composition.
        
        This method calculates the number of shares for each asset based on
        the optimization policy and the total portfolio value.
        """
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
        """Get the list of symbols in the portfolio.
        
        Returns:
            list[str]: List of symbols in the portfolio
        """
        return self.policy.portfolio_symbols

    @property
    def weights(self) -> Annotated[NDArray[np.float64], Literal["1D array"]]:
        """Get the optimized weights for each asset.
        
        Returns:
            Annotated[NDArray[np.float64], Literal["1D array"]]: Array of optimized weights
        """
        return self.policy.weights

    @property
    def portfolio_yield(self) -> float:
        """Get the expected yield of the optimized portfolio.
        
        Returns:
            float: Expected yield of the portfolio
        """
        return self.policy.portfolio_yield

    @property
    def volatility(self) -> float:
        """Get the volatility of the optimized portfolio.
        
        Returns:
            float: Volatility of the portfolio
        """
        return self.policy.volatility

    @property
    def correlation_matrix(self) -> Annotated[NDArray[np.float64], Literal["2D array"]]:
        """Get the correlation matrix of the optimized portfolio.
        
        Returns:
            Annotated[NDArray[np.float64], Literal["2D array"]]: Correlation matrix of the portfolio
        """
        return self.policy.correlation_matrix

    @property
    def named_correlation_matrix(self) -> pd.DataFrame:
        """Get the named correlation matrix of the optimized portfolio.
        
        Returns:
            pd.DataFrame: Named correlation matrix of the portfolio
        """
        return self.policy.named_correlation_matrix

    @property
    def portfolio_summary(self) -> dict[str, Any]:
        """Get a summary of the optimized portfolio.
        
        Returns:
            dict[str, Any]: Dictionary containing portfolio summary information
        """
        return self.summary

    def get_portfolio(self) -> Portfolio:
        """Get the underlying Portfolio object.
        
        Returns:
            Portfolio: The underlying Portfolio object
        """
        return Portfolio(self.symbols_nbshares, cacheddir=self.cacheddir)

    def save_to_json(self, fileobj: TextIOWrapper) -> None:
        """Save the optimized portfolio to a JSON file.
        
        Args:
            fileobj: File object to write the portfolio data to
        """
        self.get_portfolio().save_to_json(fileobj)

    def dumps_json(self) -> str:
        """Serialize the optimized portfolio to a JSON string.
        
        Returns:
            str: JSON string representation of the portfolio
        """
        return self.get_portfolio().dumps_json()

    @classmethod
    def load_from_json(
            cls,
            fileobj: TextIOWrapper,
            cacheddir: Optional[PathLike | str]=None
    ) -> Self:
        """Load an optimized portfolio from a JSON file.
        
        Note: This method is not implemented for OptimizedPortfolio. Use Portfolio.load_from_json instead.
        
        Args:
            fileobj: File object to read the portfolio data from
            cacheddir: Directory for cached data (optional)
            
        Raises:
            NotImplementedError: Always raised as this method is not implemented
        """
        raise NotImplementedError('OptimizedPortfolio does not implement loading from json. Use Portfolio to load instead.')

    @classmethod
    def load_from_dict(
            cls,
            portdict: TextIOWrapper,
            cacheddir: Optional[PathLike | str]=None
    ) -> Self:
        """Load an optimized portfolio from a dictionary.
        
        Note: This method is not implemented for OptimizedPortfolio. Use Portfolio.load_from_dict instead.
        
        Args:
            portdict: Dictionary mapping symbols to number of shares
            cacheddir: Directory for cached data (optional)
            
        Raises:
            NotImplementedError: Always raised as this method is not implemented
        """
        raise NotImplementedError('OptimizedPortfolio does not implement loading from json. Use Portfolio to load instead.')
