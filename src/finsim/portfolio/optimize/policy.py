
from itertools import product
from abc import ABC, abstractmethod
from typing import Any, Optional, Literal, Annotated

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import numba as nb

from .numerics import optimized_portfolio_on_sharperatio, optimized_portfolio_mpt_costfunction, optimized_portfolio_mpt_entropy_costfunction


@nb.njit
def mat_to_list(mat: Annotated[NDArray[np.float64], Literal["2D array"]]) -> list[list[float]]:
    """Convert a 2D numpy array to a list of lists.
    
    Args:
        mat: 2D numpy array to convert
        
    Returns:
        list[list[float]]: List of lists representation of the matrix
    """
    return [
        [
            float(mat[i, j]) for j in range(mat.shape[1])
        ]
        for i in range(mat.shape[0])
    ]


class OptimizedWeightingPolicy(ABC):
    """Abstract base class for optimized weighting policies.
    
    This class defines the interface for optimization policies used in portfolio
    optimization, including methods for calculating weights, yields, and volatilities.
    """
    
    def __init__(
            self,
            rf: float,
            r: Optional[Annotated[NDArray[np.float64], Literal["1D array"]]]=None,
            cov: Optional[Annotated[NDArray[np.float64], Literal["2D array"]]]=None,
            symbols: Optional[list[str]]=None
    ):
        """Initialize the OptimizedWeightingPolicy.
        
        Args:
            rf: Risk-free rate
            r: Array of expected returns (optional)
            cov: Covariance matrix (optional)
            symbols: List of symbols (optional)
        """
        self.rf = rf

        assert len(r) == cov.shape[0]
        assert cov.shape[0] == cov.shape[1]
        if symbols is not None:
            assert len(r) == len(symbols)

        self.r = r
        self.cov = cov
        self.symbols = symbols if symbols is not None else list(range(len(r)))

        self.optimized = False

    @abstractmethod
    def optimize(
            self,
            r: Annotated[NDArray[np.float64], Literal["1D array"]],
            cov: Annotated[NDArray[np.float64], Literal["2D array"]],
            symbols: Optional[list[str]]=None
    ) -> None:
        """Optimize the portfolio weights.
        
        Args:
            r: Array of expected returns
            cov: Covariance matrix
            symbols: List of symbols (optional)
        """
        raise NotImplemented()

    @property
    def portfolio_symbols(self) -> list[str]:
        """Get the list of symbols in the portfolio.
        
        Returns:
            list[str]: List of symbols in the portfolio
        """
        return self.symbols

    @property
    @abstractmethod
    def weights(self) -> Annotated[NDArray[np.float64], Literal["1D array"]]:
        """Get the optimized weights for each asset.
        
        Returns:
            Annotated[NDArray[np.float64], Literal["1D array"]]: Array of optimized weights
        """
        raise NotImplemented()

    @property
    @abstractmethod
    def portfolio_yield(self) -> float:
        """Get the expected yield of the optimized portfolio.
        
        Returns:
            float: Expected yield of the portfolio
        """
        raise NotImplemented()

    @property
    @abstractmethod
    def volatility(self) -> float:
        """Get the volatility of the optimized portfolio.
        
        Returns:
            float: Volatility of the portfolio
        """
        raise NotImplemented()

    @property
    def correlation_matrix(self) -> Annotated[NDArray[np.float64], Literal["2D array"]]:
        """Get the correlation matrix of the optimized portfolio.
        
        Returns:
            Annotated[NDArray[np.float64], Literal["2D array"]]: Correlation matrix of the portfolio
        """
        corr = np.zeros(self.cov.shape)
        for i, j in product(range(self.cov.shape[0]), range(self.cov.shape[1])):
            corr[i, j] = self.cov[i, j] / np.sqrt(self.cov[i, i] * self.cov[j, j])
        return corr

    @property
    def named_correlation_matrix(self) -> pd.DataFrame:
        """Get the named correlation matrix of the optimized portfolio.
        
        Returns:
            pd.DataFrame: Named correlation matrix of the portfolio
        """
        corrdf = pd.DataFrame(self.correlation_matrix,
                              columns=self.symbols,
                              index=self.symbols)
        return corrdf

    @property
    def portfolio_summary(self) -> dict[str, Any]:
        """Get a summary of the optimized portfolio.
        
        Returns:
            dict[str, Any]: Dictionary containing portfolio summary information
        """
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
            'correlation': self.correlation_matrix
        }
        return summary

    @property
    def policytype(self) -> str:
        """Get the type of the optimization policy.
        
        Returns:
            str: Type of the optimization policy
        """
        return 'AbstractOptimizedWeightingPolicy'



class OptimizedWeightingPolicyUsingMPTSharpeRatio(OptimizedWeightingPolicy):
    """Optimized weighting policy using the Sharpe ratio from Modern Portfolio Theory.
    
    This class implements portfolio optimization based on maximizing the Sharpe ratio,
    which is the ratio of excess return to volatility.
    """
    
    def __init__(
            self,
            rf: float,
            r: Optional[Annotated[NDArray[np.float64], Literal["1D array"]]]=None,
            cov: Optional[Annotated[NDArray[np.float64], Literal["2D array"]]]=None,
            symbols: Optional[list[str]]=None,
            minweight: float=0.
    ):
        """Initialize the OptimizedWeightingPolicyUsingMPTSharpeRatio.
        
        Args:
            rf: Risk-free rate
            r: Array of expected returns (optional)
            cov: Covariance matrix (optional)
            symbols: List of symbols (optional)
            minweight: Minimum weight for each asset (default: 0.0)
        """
        super(OptimizedWeightingPolicyUsingMPTSharpeRatio, self).__init__(rf, r=r, cov=cov, symbols=symbols)
        self.minweight = minweight
        if r is not None and cov is not None:
            self.optimize(r, cov, symbols=symbols)

    def optimize(
            self,
            r: Annotated[NDArray[np.float64], Literal["1D array"]],
            cov: Annotated[NDArray[np.float64], Literal["2D array"]],
            symbols: Optional[list[str]]=None
    ) -> None:
        """Optimize the portfolio weights using the Sharpe ratio.
        
        Args:
            r: Array of expected returns
            cov: Covariance matrix
            symbols: List of symbols (optional)
        """
        self.optimized_sol = optimized_portfolio_on_sharperatio(r, cov, self.rf, minweight=self.minweight)
        self.optimized = True

        self.optimized_weights = self.optimized_sol.x
        self.optimized_sharpe_ratio = -self.optimized_sol.fun
        self.optimized_portfolio_yield = np.sum(self.optimized_weights * self.r)
        sqweights = np.dot(
            np.expand_dims(self.optimized_weights, axis=1),
            np.expand_dims(self.optimized_weights, axis=0)
        )
        self.optimized_volatility = np.sqrt(np.sum(sqweights * self.cov))

    @property
    def weights(self) -> Annotated[NDArray[np.float64], Literal["1D array"]]:
        """Get the optimized weights for each asset.
        
        Returns:
            Annotated[NDArray[np.float64], Literal["1D array"]]: Array of optimized weights
        """
        return self.optimized_weights

    @property
    def portfolio_yield(self) -> float:
        """Get the expected yield of the optimized portfolio.
        
        Returns:
            float: Expected yield of the portfolio
        """
        return self.optimized_portfolio_yield

    @property
    def volatility(self) -> float:
        """Get the volatility of the optimized portfolio.
        
        Returns:
            float: Volatility of the portfolio
        """
        return self.optimized_volatility

    @property
    def sharpe_ratio(self) -> float:
        """Get the Sharpe ratio of the optimized portfolio.
        
        Returns:
            float: Sharpe ratio of the portfolio
        """
        return self.optimized_sharpe_ratio

    @property
    def portfolio_summary(self) -> dict[str, Any]:
        """Get a summary of the optimized portfolio.
        
        Returns:
            dict[str, Any]: Dictionary containing portfolio summary information
        """
        summary = super(OptimizedWeightingPolicyUsingMPTSharpeRatio, self).portfolio_summary
        summary['sharpe_ratio'] = self.optimized_sharpe_ratio
        return summary

    @property
    def policytype(self) -> str:
        """Get the type of the optimization policy.
        
        Returns:
            str: Type of the optimization policy
        """
        return 'OptimizedWeightingPolicyUsingMPTSharpeRatio'


class OptimizedWeightingPolicyUsingMPTCostFunction(OptimizedWeightingPolicy):
    """Optimized weighting policy using the MPT cost function.
    
    This class implements portfolio optimization based on a cost function
    from Modern Portfolio Theory.
    """
    
    def __init__(
            self,
            rf: float,
            r: Annotated[NDArray[np.float64], Literal["1D array"]],
            cov: Annotated[NDArray[np.float64], Literal["2D array"]],
            symbols: Optional[list[str]]=None,
            lamb: float=0.0,
            V0: float=10.0
    ):
        """Initialize the OptimizedWeightingPolicyUsingMPTCostFunction.
        
        Args:
            rf: Risk-free rate
            r: Array of expected returns
            cov: Covariance matrix
            symbols: List of symbols (optional)
            lamb: Lambda parameter for the cost function (optional)
            V0: Initial portfolio value (default: 10.0)
        """
        super(OptimizedWeightingPolicyUsingMPTCostFunction, self).__init__(rf, r=r, cov=cov, symbols=symbols)
        self.lamb = lamb
        self.V0 = V0

        if r is not None and cov is not None:
            self.optimize(r, cov, symbols=symbols)

    def optimize(
            self,
            r: Annotated[NDArray[np.float64], Literal["1D array"]],
            cov: Annotated[NDArray[np.float64], Literal["2D array"]],
            symbols: Optional[list[str]]=None
    ) -> None:
        """Optimize the portfolio weights using the MPT cost function.
        
        Args:
            r: Array of expected returns
            cov: Covariance matrix
            symbols: List of symbols (optional)
        """
        self.optimized_sol = optimized_portfolio_mpt_costfunction(r, cov, self.rf, self.lamb, V0=self.V0)
        self.optimized = True

        self.optimized_unnormalized_weights = self.optimized_sol.x
        self.optimized_weights = self.optimized_unnormalized_weights[:-1] / np.sum(self.optimized_unnormalized_weights[:-1])
        self.optimized_costfunction = -self.optimized_sol.fun
        self.optimized_portfolio_yield = np.sum(self.optimized_weights * self.r)
        sqweights = np.dot(
            np.expand_dims(self.optimized_weights, axis=1),
            np.expand_dims(self.optimized_weights, axis=0)
        )
        self.optimized_volatility = np.sqrt(np.sum(sqweights * self.cov))

    @property
    def weights(self) -> Annotated[NDArray[np.float64], Literal["1D array"]]:
        """Get the optimized weights for each asset.
        
        Returns:
            Annotated[NDArray[np.float64], Literal["1D array"]]: Array of optimized weights
        """
        return self.optimized_weights

    @property
    def portfolio_yield(self) -> float:
        """Get the expected yield of the optimized portfolio.
        
        Returns:
            float: Expected yield of the portfolio
        """
        return self.optimized_portfolio_yield

    @property
    def volatility(self) -> float:
        """Get the volatility of the optimized portfolio.
        
        Returns:
            float: Volatility of the portfolio
        """
        return self.optimized_volatility

    @property
    def mpt_costfunction(self) -> float:
        """Get the MPT cost function value of the optimized portfolio.
        
        Returns:
            float: MPT cost function value of the portfolio
        """
        return self.optimized_costfunction

    @property
    def portfolio_summary(self) -> dict[str, Any]:
        """Get a summary of the optimized portfolio.
        
        Returns:
            dict[str, Any]: Dictionary containing portfolio summary information
        """
        summary = super(OptimizedWeightingPolicyUsingMPTCostFunction, self).portfolio_summary
        summary['mpt_costfunction'] = self.mpt_costfunction
        summary['V0'] = self.V0
        summary['lamb'] = self.lamb
        return summary

    @property
    def policytype(self) -> str:
        """Get the type of the optimization policy.
        
        Returns:
            str: Type of the optimization policy
        """
        return 'OptimizedWeightingPolicyUsingMPTCostFunction'


class OptimizedWeightingPolicyUsingMPTEntropyCostFunction(OptimizedWeightingPolicy):
    """Optimized weighting policy using the MPT entropy cost function.
    
    This class implements portfolio optimization based on an entropy cost function
    from Modern Portfolio Theory.
    """
    
    def __init__(
            self,
            rf: float,
            r: Annotated[NDArray[np.float64], Literal["1D array"]]=None,
            cov: Annotated[NDArray[np.float64], Literal["2D array"]]=None,
            symbols: Optional[list[str]]=None,
            lamb0: float=0.0,
            lamb1: float=0.0,
            V: float=10.0
    ):
        """Initialize the OptimizedWeightingPolicyUsingMPTEntropyCostFunction.
        
        Args:
            rf: Risk-free rate
            r: Array of expected returns (optional)
            cov: Covariance matrix (optional)
            symbols: List of symbols (optional)
            lamb0: Lambda 0 parameter for the entropy cost function (optional)
            lamb1: Lambda 1 parameter for the entropy cost function (optional)
            V: Portfolio value parameter (default: 10.0)
        """
        super(OptimizedWeightingPolicyUsingMPTEntropyCostFunction, self).__init__(rf, r=r, cov=cov, symbols=symbols)
        self.lamb0 = lamb0
        self.lamb1 = lamb1
        self.V = V

        if r is not None and cov is not None:
            self.optimize(r, cov, symbols=symbols)

    def optimize(
            self,
            r: Annotated[NDArray[np.float64], Literal["1D array"]],
            cov: Annotated[NDArray[np.float64], Literal["2D array"]],
            symbols: Optional[list[str]]=None
    ) -> None:
        """Optimize the portfolio weights using the MPT entropy cost function.
        
        Args:
            r: Array of expected returns
            cov: Covariance matrix
            symbols: List of symbols (optional)
        """
        self.optimized_sol = optimized_portfolio_mpt_entropy_costfunction(r, cov, self.rf, self.lamb0, self.lamb1, V=self.V)
        self.optimized = True

        self.optimized_unnormalized_weights = self.optimized_sol.x
        self.optimized_weights = self.optimized_unnormalized_weights[:-1] / np.sum(self.optimized_unnormalized_weights[:-1])
        self.optimized_costfunction = -self.optimized_sol.fun
        self.optimized_portfolio_yield = np.sum(self.optimized_weights * self.r)
        sqweights = np.dot(
            np.expand_dims(self.optimized_weights, axis=1),
            np.expand_dims(self.optimized_weights, axis=0)
        )
        self.optimized_volatility = np.sqrt(np.sum(sqweights * self.cov))

    @property
    def weights(self) -> Annotated[NDArray[np.float64], Literal["1D array"]]:
        """Get the optimized weights for each asset.
        
        Returns:
            Annotated[NDArray[np.float64], Literal["1D array"]]: Array of optimized weights
        """
        return self.optimized_weights

    @property
    def portfolio_yield(self) -> float:
        """Get the expected yield of the optimized portfolio.
        
        Returns:
            float: Expected yield of the portfolio
        """
        return self.optimized_portfolio_yield

    @property
    def volatility(self) -> float:
        """Get the volatility of the optimized portfolio.
        
        Returns:
            float: Volatility of the portfolio
        """
        return self.optimized_volatility

    @property
    def mpt_entropy_costfunction(self) -> float:
        """Get the MPT entropy cost function value of the optimized portfolio.
        
        Returns:
            float: MPT entropy cost function value of the portfolio
        """
        return self.optimized_costfunction

    @property
    def portfolio_summary(self) -> dict[str, Any]:
        """Get a summary of the optimized portfolio.
        
        Returns:
            dict[str, Any]: Dictionary containing portfolio summary information
        """
        summary = super(OptimizedWeightingPolicyUsingMPTEntropyCostFunction, self).portfolio_summary
        summary['mpt_entropy_costfunction'] = self.mpt_entropy_costfunction
        summary['V'] = self.V
        summary['lamb0'] = self.lamb0
        summary['lamb1'] = self.lamb1
        return summary

    @property
    def policytype(self) -> str:
        """Get the type of the optimization policy.
        
        Returns:
            str: Type of the optimization policy
        """
        return 'OptimizedWeightingPolicyUsingMPTEntropyCostFunction'
