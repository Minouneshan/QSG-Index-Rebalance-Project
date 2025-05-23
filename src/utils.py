import numpy as np

def sharpe_ratio(returns, risk_free=0.0):
    """Compute annualized Sharpe Ratio."""    
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if returns.std() == 0: return 0
    return (returns.mean() - risk_free) / returns.std() * np.sqrt(252)

def max_drawdown(pnl_curve):
    """Compute max drawdown (peak-to-trough)."""
    pnl_array = np.array(pnl_curve)
    running_max = np.maximum.accumulate(pnl_array)
    drawdowns = running_max - pnl_array
    return np.max(drawdowns)

def sortino_ratio(returns, risk_free=0.0, periods_per_year=252):
    downside = returns[returns < 0]
    downside_std = np.std(downside)
    mean = np.nanmean(returns) - risk_free/periods_per_year
    if downside_std == 0:
        return np.nan
    return mean / downside_std * np.sqrt(periods_per_year)


def win_rate(pnl):
    pnl = np.array(pnl)
    return np.mean(pnl > 0)

