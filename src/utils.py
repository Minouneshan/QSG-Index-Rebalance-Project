# src/utils.py

def sharpe_ratio(returns, risk_free=0.0):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if returns.std() == 0: return 0
    return (returns.mean() - risk_free) / returns.std() * np.sqrt(252)

def max_drawdown(pnl_curve):
    pnl_array = np.array(pnl_curve)
    running_max = np.maximum.accumulate(pnl_array)
    drawdowns = running_max - pnl_array
    return np.max(drawdowns)
