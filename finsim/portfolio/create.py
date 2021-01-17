
from .optimize.numerics import get_BlackScholesMerton_stocks_estimation
from .portfolio import OptimizedPortfolio
from .optimize.policy import OptimizedWeightingPolicyUsingMPTSharpeRatio, \
    OptimizedWeightingPolicyUsingMPTCostFunction, OptimizedWeightingPolicyUsingMPTEntropyCostFunction


def get_optimized_portfolio_on_sharpe_ratio(
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
    optimized_portfolio = OptimizedPortfolio(optimized_weighting_policy, totalworth, presetdate, cacheddir=cacheddir)
    return optimized_portfolio


def get_optimized_portfolio_on_mpt_costfunction(
        rf,
        symbols,
        totalworth,
        presetdate,
        estimating_startdate,
        estimating_enddate,
        lamb,
        V0=10.,
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
    optimized_weighting_policy = OptimizedWeightingPolicyUsingMPTCostFunction(rf, r, cov, symbols, lamb, V0=V0)
    optimized_portfolio = OptimizedPortfolio(optimized_weighting_policy, totalworth, presetdate, cacheddir=cacheddir)
    return optimized_portfolio


def get_optimized_portfolio_on_mpt_entropy_costfunction(
        rf,
        symbols,
        totalworth,
        presetdate,
        estimating_startdate,
        estimating_enddate,
        lamb0,
        lamb1,
        V=10.,
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
    optimized_weighting_policy = OptimizedWeightingPolicyUsingMPTEntropyCostFunction(rf, r, cov, symbols, lamb0, lamb1, V=V)
    optimized_portfolio = OptimizedPortfolio(optimized_weighting_policy, totalworth, presetdate, cacheddir=cacheddir)
    return optimized_portfolio