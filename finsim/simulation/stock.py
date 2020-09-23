from abc import ABC, abstractmethod
from math import log, exp

import numpy as np


class AbstractStochasticValue(ABC):
    @abstractmethod
    def generate_time_series(self, T, dt, nbsimulations=1):
        pass


class BlackScholesMertonStockPrices(AbstractStochasticValue):
    def __init__(self, S0, r, sigma):
        self.S0 = S0
        self.r = r
        self.sigma = sigma

        self.logS0 = log(S0)

    def generate_time_series(self, T, dt, nbsimulations=1):
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
    def __init__(self, x0, theta, kappa, sigma):
        self.x0 = x0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma

    def generate_time_series(self, T, dt, nbsimulations=1):
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
    def __init__(self, S0, r, v0, theta, kappa, sigma_v, rho):
        self.S0 = S0
        self.r = r
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma_v = sigma_v
        self.rho = rho

        self.logS0 = log(self.S0)
        self.rho = np.array([[1., self.rho], [self.rho, 1.]])

    def generate_time_series(self, T, dt, nbsimulations=1):
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
                         (self.r - 0.5 * v[:, i] * v[:, i]) * dt + \
                         v[:, i] * z1[:, i] * np.sqrt(dt)

        return np.exp(logS), v


class MertonJumpDiffusionStockPrices(AbstractStochasticValue):
    def __init__(self, S0, r, sigma, mu, lamb, delta):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.mu = mu
        self.lamb = lamb
        self.delta = delta

        self.logS0 = log(self.S0)

    def generate_time_series(self, T, dt, nbsimulations=1):
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
