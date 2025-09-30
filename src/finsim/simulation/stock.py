
from abc import ABC, abstractmethod
from math import log, exp
from typing import Literal, Annotated

import numpy as np
from numpy.typing import NDArray


class AbstractStochasticValue(ABC):
    """Abstract base class for stochastic value generators."""
    
    @abstractmethod
    def generate_time_series(self, T: float, dt: float, nbsimulations: int=1):
        """Generate a time series of values.
        
        Args:
            T: Time horizon
            dt: Time step
            nbsimulations: Number of simulations (default: 1)
        """
        raise NotImplemented()


class BlackScholesMertonStockPrices(AbstractStochasticValue):
    """Generate stock prices using the Black-Scholes-Merton model.
    
    This class implements the Black-Scholes-Merton model for stock price simulation.
    """
    
    def __init__(self, S0: float, r: float, sigma: float):
        """Initialize the Black-Scholes-Merton stock price generator.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma

        self.logS0 = log(S0)

    def generate_time_series(
            self,
            T: float,
            dt: float,
            nbsimulations: int=1
    ) -> Annotated[NDArray[np.float64], Literal["1D array"]]:
        """Generate a time series of stock prices using the Black-Scholes-Merton model.
        
        Args:
            T: Time horizon
            dt: Time step
            nbsimulations: Number of simulations (default: 1)
            
        Returns:
            NDArray[Shape["*"], Float]: Array of stock prices
        """
        nbtimesteps = int(T // dt) + 1
        z = np.random.normal(size=(nbsimulations, nbtimesteps))
        logS = np.zeros((nbsimulations, nbtimesteps))
        logS[:, 0] = self.logS0
        for i in range(1, nbtimesteps):
            logS[:, i] = logS[:, i-1] + \
                         (self.r - 0.5 * self.sigma * self.sigma) * dt + \
                         self.sigma * z[:, i] * np.sqrt(dt)
        return np.exp(logS)


class SquareRootDiffusionProcesses(AbstractStochasticValue):
    """Generate values using the square root diffusion process.
    
    This class implements the square root diffusion process for value simulation.
    """
    
    def __init__(self, x0: float, theta: float, kappa: float, sigma: float):
        """Initialize the square root diffusion process generator.
        
        Args:
            x0: Initial value
            theta: Long-term mean
            kappa: Speed of mean reversion
            sigma: Volatility
        """
        self.x0 = x0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma

    def generate_time_series(
            self,
            T: float,
            dt: float,
            nbsimulations: int=1
    ) -> Annotated[NDArray[np.float64], Literal["1D array"]]:
        """Generate a time series of values using the square root diffusion process.
        
        Args:
            T: Time horizon
            dt: Time step
            nbsimulations: Number of simulations (default: 1)
            
        Returns:
            NDArray[Shape["*"], Float]: Array of values
        """
        nbtimesteps = int(T // dt) + 1
        z = np.random.normal(size=(nbsimulations, nbtimesteps))
        xarray = np.zeros((nbsimulations, nbtimesteps))
        xarray[:, 0] = self.x0
        for i in range(1, nbtimesteps):
            xarray[:, i] = xarray[:, i - 1] + \
                           self.kappa * (self.theta - xarray[:, i - 1]) * dt + \
                           self.sigma * np.sqrt(xarray[:, i - 1]) * z[:, i] * np.sqrt(dt)
        return xarray


class HestonStockPrices(AbstractStochasticValue):
    """Generate stock prices using the Heston model.
    
    This class implements the Heston model for stock price simulation with stochastic volatility.
    """
    
    def __init__(self, S0: float, r: float, v0: float, theta: float, kappa: float, sigma_v: float, rho: float):
        """Initialize the Heston stock price generator.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            v0: Initial variance
            theta: Long-term variance
            kappa: Speed of mean reversion
            sigma_v: Volatility of variance
            rho: Correlation between stock price and variance
        """
        self.S0 = S0
        self.r = r
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma_v = sigma_v
        self.rho = rho

        self.logS0 = log(self.S0)
        self.rho = np.array([[1., self.rho], [self.rho, 1.]])

    def generate_time_series(self, T: float, dt: float, nbsimulations: int=1) -> Annotated[NDArray[np.float64], Literal["1D array"]]:
        """Generate a time series of stock prices using the Heston model.
        
        Args:
            T: Time horizon
            dt: Time step
            nbsimulations: Number of simulations (default: 1)
            
        Returns:
            NDArray[Shape["*"], Float]: Array of stock prices
        """
        nbtimesteps = int(T // dt) + 1

        # generate correlated random numbers
        Z = np.random.multivariate_normal((0., 0.), self.rho, size=(nbsimulations, nbtimesteps))
        z1 = Z[:, :, 0]
        z2 = Z[:, :, 1]

        # stochastic volatility
        v = np.zeros((nbsimulations, nbtimesteps))
        v[:, 0] = self.v0
        for i in range(1, nbtimesteps):
            v[:, i] = v[:, i - 1] + \
                      self.kappa * (self.theta - v[:, i - 1]) * dt + \
                      self.sigma_v * np.sqrt(v[:, i - 1]) * z2[:, i] * np.sqrt(dt)

        # stock price
        logS = np.zeros((nbsimulations, nbtimesteps))
        logS[:, 0] = self.logS0
        for i in range(1, nbtimesteps):
            logS[:, i] = logS[:, i-1] + \
                         (self.r - 0.5 * v[:, i]) * dt + \
                         np.sqrt(v[:, i]) * z1[:, i] * np.sqrt(dt)

        return np.exp(logS), v


class MertonJumpDiffusionStockPrices(AbstractStochasticValue):
    """Generate stock prices using the Merton jump-diffusion model.
    
    This class implements the Merton jump-diffusion model for stock price simulation.
    """
    
    def __init__(self, S0: float, r: float, sigma: float, mu: float, lamb: float, delta: float):
        """Initialize the Merton jump-diffusion stock price generator.
        
        Args:
            S0: Initial stock price
            r: Risk-free rate
            sigma: Volatility
            mu: Expected jump size
            lamb: Jump intensity
            delta: Jump size volatility
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.mu = mu
        self.lamb = lamb
        self.delta = delta

        self.logS0 = log(self.S0)

    def generate_time_series(self, T: float, dt: float, nbsimulations: int=1) -> Annotated[NDArray[np.float64], Literal["1D array"]]:
        """Generate a time series of stock prices using the Merton jump-diffusion model.
        
        Args:
            T: Time horizon
            dt: Time step
            nbsimulations: Number of simulations (default: 1)
            
        Returns:
            NDArray[Shape["*"], Float]: Array of stock prices
        """
        nbtimesteps = int(T // dt) + 1
        z1 = np.random.normal(size=(nbsimulations, nbtimesteps))
        logS = np.zeros((nbsimulations, nbtimesteps))
        logS[:, 0] = self.logS0
        rJ = self.lamb * (exp(self.mu+0.5*self.delta*self.delta) - 1)
        nbjumps = np.random.poisson(self.lamb*dt, (nbsimulations, nbtimesteps))
        jumpmagnitudes = np.random.lognormal(
            log(1+self.mu)-0.5*self.delta*self.delta,
            self.delta,
            size=(nbsimulations, nbtimesteps)
        )
        for i in range(1, nbtimesteps):
            logS[:, i] = logS[:, i-1] + \
                         (self.r - rJ - 0.5 * self.sigma * self.sigma) * dt + \
                         self.sigma * z1[:, i] * np.sqrt(dt) + \
                         jumpmagnitudes[:, i] * nbjumps[:, i]
        return np.exp(logS)
