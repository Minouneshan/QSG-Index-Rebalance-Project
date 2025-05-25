"""
utils.py

This module provides utility functions for performance metrics, logging, and summary saving in the QSG Index Rebalance Project.

Functionality:
- Implements Sharpe ratio, Sortino ratio, max drawdown, win rate, and other performance metrics.
- Provides robust summary saving and logging utilities for reporting and reproducibility.
- Designed for use by all strategy and backtest modules.

Readability:
- All functions are self-contained and use clear, descriptive names and docstrings.
- Designed for clarity, extensibility, and easy integration with other modules.

This file is part of a modular pipeline, with supporting strategy code in strategies.py and orchestration in backtest_strategies.py.
"""

import numpy as np
import json
import os

# --- Performance Metrics ---

def sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the annualized Sharpe ratio of a return series.

    Args:
        returns (pd.Series): Series of returns.
        risk_free_rate (float): Annual risk-free rate (default 0.0).

    Returns:
        float: Annualized Sharpe ratio.
    """    
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if returns.std() == 0: return 0
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)

def sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the annualized Sortino ratio of a return series.

    Args:
        returns (pd.Series): Series of returns.
        risk_free_rate (float): Annual risk-free rate (default 0.0).

    Returns:
        float: Annualized Sortino ratio.
    """
    downside = returns[returns < 0]
    downside_std = np.std(downside)
    mean = np.nanmean(returns) - risk_free_rate/252
    if downside_std == 0:
        return np.nan
    return mean / downside_std * np.sqrt(252)

def max_drawdown(pnl_curve):
    """
    Calculate the maximum drawdown of a cumulative P&L or equity curve.

    Args:
        pnl_curve (pd.Series): Cumulative P&L or equity curve.

    Returns:
        float: Maximum drawdown value.
    """
    pnl_array = np.array(pnl_curve)
    running_max = np.maximum.accumulate(pnl_array)
    drawdowns = running_max - pnl_array
    return np.max(drawdowns)

def win_rate(pnl):
    """
    Calculate the win rate (fraction of positive trades) for a P&L series.

    Args:
        pnl (pd.Series): Series of trade P&L values.

    Returns:
        float: Win rate (between 0 and 1).
    """
    pnl = np.array(pnl)
    return np.mean(pnl > 0)

def save_summary(summary, name):
    """
    Save summary statistics or results to a JSON file for reporting and reproducibility.

    Args:
        summary (dict): Summary statistics or results.
        name (str): Name for the output file (used in filename).

    Returns:
        None
    """
    os.makedirs("results/summaries", exist_ok=True)
    with open(f"results/summaries/{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)