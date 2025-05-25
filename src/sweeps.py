"""
sweeps.py

This module implements parameter sweep utilities for the QSG Index Rebalance Project.

Functionality:
- Provides functions to run systematic sweeps over strategy parameters (e.g., holding period, thresholds).
- Designed for integration with strategy and backtest modules for robust parameter analysis.

Readability:
- All functions are self-contained and use clear, descriptive names and docstrings.
- Designed for clarity, extensibility, and easy integration with other modules.

This file is part of a modular pipeline, with supporting strategy code in strategies.py, utilities in utils.py, and orchestration in backtest_strategies.py.
"""

from strategies import post_announcement_momentum, event_day_reversion, buy_and_hold, hedged_momentum
from utils import save_summary


def sweep_post_announcement(events, price, hold_periods, **kwargs):
    """
    Sweep over holding periods for the post-announcement momentum strategy.

    Args:
        events (pd.DataFrame): Event data.
        price (pd.DataFrame): Price data.
        hold_periods (list): List of holding periods to test.
        **kwargs: Additional keyword arguments for the strategy function.

    Returns:
        dict: Dictionary of results for each holding period.
    """
    sweep_metrics = {}
    for N in hold_periods:
        bt, summary = post_announcement_momentum(
            events, price,
            hold_days=N,
            **kwargs
        )
        sweep_metrics[str(N)] = summary  # Ensure key is a string
    save_summary(sweep_metrics, "post_announcement_momentum_sweep")
    return sweep_metrics



def sweep_event_day_reversion(events, price, thresholds, hold_periods, **kwargs):
    """
    Sweep over threshold and holding period combinations for the event day reversion strategy.

    Args:
        events (pd.DataFrame): Event data.
        price (pd.DataFrame): Price data.
        thresholds (list): List of threshold values to test.
        hold_periods (list): List of holding periods to test.
        **kwargs: Additional keyword arguments for the strategy function.

    Returns:
        dict: Dictionary of results for each threshold and holding period combination.
    """
    sweep_metrics = {}
    for th in thresholds:
        for N in hold_periods:
            rv, summary = event_day_reversion(
                events, price,
                threshold=th, hold_days=N,  # <-- Add hold_days param
                **kwargs
            )
            sweep_metrics[f"{th}_{N}"] = summary  # Use string key
    save_summary(sweep_metrics, "event_day_reversion_sweep")
    return sweep_metrics



def sweep_buy_and_hold(events, price, entry_lags, hold_periods, **kwargs):
    """
    Sweep over entry lag and holding period combinations for the buy and hold strategy.

    Args:
        events (pd.DataFrame): Event data.
        price (pd.DataFrame): Price data.
        entry_lags (list): List of entry lags to test.
        hold_periods (list): List of holding periods to test.
        **kwargs: Additional keyword arguments for the strategy function.

    Returns:
        dict: Dictionary of results for each entry lag and holding period combination.
    """
    sweep_metrics = {}
    for lag in entry_lags:
        for N in hold_periods:
            bh, summary = buy_and_hold(
                events, price,
                entry_lag=lag, hold_days=N,
                **kwargs
            )
            sweep_metrics[f"{lag}_{N}"] = summary  # Use string key
    save_summary(sweep_metrics, "buy_and_hold_sweep")
    return sweep_metrics


def sweep_hedged_momentum(
    events, price, spy,
    hedge_ratios=[1.0],
    hold_periods=[5],
    min_advs=[1],
    **kwargs
):
    """
    Sweep over hedge ratio, holding period, and minimum ADV combinations for the hedged momentum strategy.

    Args:
        events (pd.DataFrame): Event data.
        price (pd.DataFrame): Price data.
        spy (pd.DataFrame): Benchmark data.
        hedge_ratios (list): List of hedge ratios to test.
        hold_periods (list): List of holding periods to test.
        min_advs (list): List of minimum ADV values to test.
        **kwargs: Additional keyword arguments for the strategy function.

    Returns:
        dict: Dictionary of results for each combination of hedge ratio, holding period, and minimum ADV.
    """
    sweep_metrics = {}
    for hr in hedge_ratios:
        for N in hold_periods:
            for ma in min_advs:
                df, summary = hedged_momentum(
                    events, price, spy,
                    hedge_ratio=hr,
                    hold_days=N,
                    min_adv=ma,
                    **kwargs
                )
                sweep_metrics[f"{hr}_{N}_{ma}"] = summary  # Use string key
    save_summary(sweep_metrics, "hedged_momentum_sweep")
    return sweep_metrics


def sweep_param_grid(
    param_grid, strategy_func, events, price, spy=None, save_dir=None
):
    """
    Runs a parameter sweep for a given strategy function over a grid of parameters.
    
    Parameters:
        param_grid (dict): Dictionary where keys are parameter names and values are lists of parameter values to sweep.
        strategy_func (callable): Strategy function to evaluate.
        events (pd.DataFrame): Event data for the strategy.
        price (pd.DataFrame): Price data for the strategy.
        spy (pd.DataFrame or None): SPY data if required by the strategy.
        save_dir (str or None): Directory to save results. If None, does not save.
    
    Returns:
        dict: Mapping from parameter combinations to (results, summary) tuples.
    """
    # ...existing code...