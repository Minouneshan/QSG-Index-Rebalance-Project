"""
optimizer.py

This module provides optimization routines for parameter selection in the QSG Index Rebalance Project.

Functionality:
- Implements grid search and machine learning-based optimization for strategy parameters.
- Adapts scalar and grid parameters for compatibility with sweep and strategy functions.
- Handles empty or missing results robustly, with clear error handling and logging.
- Designed for extensibility to support additional optimization methods as needed.

Readability:
- All optimization functions are modular, well-documented, and easy to integrate with the pipeline.
- Results are saved in a structured format for downstream analysis and reporting.

This file is part of a modular pipeline, with strategies in strategies.py and sweeps in sweeps.py.
"""

import pandas as pd
import numpy as np
from sweeps import sweep_post_announcement, sweep_event_day_reversion, sweep_buy_and_hold, sweep_hedged_momentum
from utils import save_summary
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor

# Dedicated optimizer for post_announcement_momentum

def optimize_post_announcement(events, price, hold_periods, target_metric='sharpe', model=None, **kwargs):
    """
    Optimize the post-announcement momentum strategy using machine learning.

    Parameters:
    - events: DataFrame containing event data.
    - price: DataFrame containing price data.
    - hold_periods: List of holding periods to evaluate.
    - target_metric: The metric to optimize (default is 'sharpe').
    - model: Optional machine learning model for prediction.

    Returns:
    - best_params: Dictionary of the best parameters found.
    - best_score: The best score achieved with the best parameters.
    - df: DataFrame containing all sweep results.
    - model: The trained machine learning model.
    """
    sweep_results = sweep_post_announcement(events, price, hold_periods, **kwargs)
    df = pd.DataFrame.from_dict(sweep_results, orient='index')
    df = df.reset_index().rename(columns={'index': 'hold_days'})
    if df.empty or target_metric not in df.columns or df[target_metric].isnull().all():
        print(f"No valid results or '{target_metric}' metric found in sweep. Check your data and strategy filters.")
        return {}, None, df, None
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[['hold_days']].values
    y = df[target_metric].values
    model.fit(X, y)
    best_idx = np.argmax(model.predict(X))
    best_params = {'hold_days': df.iloc[best_idx]['hold_days']}
    best_score = y[best_idx]
    return best_params, best_score, df, model

# Dedicated optimizer for event_day_reversion

def optimize_event_day_reversion(events, price, thresholds, hold_periods, target_metric='sharpe', model=None, **kwargs):
    """
    Optimize the event day reversion strategy using machine learning.

    Parameters:
    - events: DataFrame containing event data.
    - price: DataFrame containing price data.
    - thresholds: List of thresholds to evaluate.
    - hold_periods: List of holding periods to evaluate.
    - target_metric: The metric to optimize (default is 'sharpe').
    - model: Optional machine learning model for prediction.

    Returns:
    - best_params: Dictionary of the best parameters found.
    - best_score: The best score achieved with the best parameters.
    - df: DataFrame containing all sweep results.
    - model: The trained machine learning model.
    """
    sweep_results = sweep_event_day_reversion(events, price, thresholds, hold_periods, **kwargs)
    df = pd.DataFrame.from_dict(sweep_results, orient='index')
    df = df.reset_index()
    df[['threshold', 'hold_days']] = pd.DataFrame(df['index'].tolist(), index=df.index)
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[['threshold', 'hold_days']].values
    y = df[target_metric].values
    model.fit(X, y)
    best_idx = np.argmax(model.predict(X))
    best_params = {'threshold': df.iloc[best_idx]['threshold'], 'hold_days': df.iloc[best_idx]['hold_days']}
    best_score = y[best_idx]
    return best_params, best_score, df, model

# Dedicated optimizer for buy_and_hold

def optimize_buy_and_hold(events, price, entry_lags, hold_periods, target_metric='sharpe', model=None, **kwargs):
    """
    Optimize the buy and hold strategy using machine learning.

    Parameters:
    - events: DataFrame containing event data.
    - price: DataFrame containing price data.
    - entry_lags: List of entry lags to evaluate.
    - hold_periods: List of holding periods to evaluate.
    - target_metric: The metric to optimize (default is 'sharpe').
    - model: Optional machine learning model for prediction.

    Returns:
    - best_params: Dictionary of the best parameters found.
    - best_score: The best score achieved with the best parameters.
    - df: DataFrame containing all sweep results.
    - model: The trained machine learning model.
    """
    sweep_results = sweep_buy_and_hold(events, price, entry_lags, hold_periods, **kwargs)
    df = pd.DataFrame.from_dict(sweep_results, orient='index')
    df = df.reset_index()
    df[['entry_lag', 'hold_days']] = pd.DataFrame(df['index'].tolist(), index=df.index)
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[['entry_lag', 'hold_days']].values
    y = df[target_metric].values
    model.fit(X, y)
    best_idx = np.argmax(model.predict(X))
    best_params = {'entry_lag': df.iloc[best_idx]['entry_lag'], 'hold_days': df.iloc[best_idx]['hold_days']}
    best_score = y[best_idx]
    return best_params, best_score, df, model

# Dedicated optimizer for hedged_momentum

def optimize_hedged_momentum(events, price, spy, hedge_ratios, hold_periods, min_advs, target_metric='sharpe', model=None, **kwargs):
    """
    Optimize the hedged momentum strategy using machine learning.

    Parameters:
    - events: DataFrame containing event data.
    - price: DataFrame containing price data.
    - spy: DataFrame containing SPY data.
    - hedge_ratios: List of hedge ratios to evaluate.
    - hold_periods: List of holding periods to evaluate.
    - min_advs: List of minimum ADVs to evaluate.
    - target_metric: The metric to optimize (default is 'sharpe').
    - model: Optional machine learning model for prediction.

    Returns:
    - best_params: Dictionary of the best parameters found.
    - best_score: The best score achieved with the best parameters.
    - df: DataFrame containing all sweep results.
    - model: The trained machine learning model.
    """
    sweep_results = sweep_hedged_momentum(events, price, spy, hedge_ratios=hedge_ratios, hold_periods=hold_periods, min_advs=min_advs, **kwargs)
    df = pd.DataFrame.from_dict(sweep_results, orient='index')
    df = df.reset_index()
    df[['hedge_ratio', 'hold_days', 'min_adv']] = pd.DataFrame(df['index'].tolist(), index=df.index)
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = df[['hedge_ratio', 'hold_days', 'min_adv']].values
    y = df[target_metric].values
    model.fit(X, y)
    best_idx = np.argmax(model.predict(X))
    best_params = {
        'hedge_ratio': df.iloc[best_idx]['hedge_ratio'],
        'hold_days': df.iloc[best_idx]['hold_days'],
        'min_adv': df.iloc[best_idx]['min_adv']
    }
    best_score = y[best_idx]
    return best_params, best_score, df, model

# General optimizer for any sweep function and param grid

def optimize_with_ml(sweep_func, param_grid, events, price, target_metric='sharpe', model=None, **kwargs):
    """
    Optimize a strategy by sweeping parameters and selecting the best result by a chosen metric.

    Parameters:
    - sweep_func: The sweep function to use for optimization.
    - param_grid: The parameter grid to search over.
    - events: The event data.
    - price: The price data.
    - target_metric: The metric to optimize (default is 'sharpe').
    - model: Optional machine learning model for prediction.

    Returns:
    - best_params: Dictionary of the best parameters found.
    - best_score: The best score achieved with the best parameters.
    - df: DataFrame containing all sweep results.
    - model: The trained machine learning model.
    """
    grid = list(ParameterGrid(param_grid))
    results = []
    for params in grid:
        sweep_result = sweep_func(events, price, **params, **kwargs)
        for k, summary in sweep_result.items():
            row = dict(params)
            if isinstance(k, tuple):
                for i, v in enumerate(k):
                    row[f'param_{i}'] = v
            else:
                row['param'] = k
            row.update(summary)
            results.append(row)
    df = pd.DataFrame(results)
    param_cols = [c for c in df.columns if c.startswith('param') or c in param_grid]
    # Robustness: Check for empty DataFrame or missing/invalid target metric
    if df.empty or target_metric not in df.columns or df[target_metric].isnull().all():
        print(f"No valid results or '{target_metric}' metric found in sweep. Check your data and strategy filters.")
        return {}, None, df, None
    X = df[param_cols].values
    y = df[target_metric].values
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    best_idx = np.argmax(model.predict(X))
    best_params = df.iloc[best_idx][param_cols].to_dict()
    best_score = y[best_idx]
    return best_params, best_score, df, model

def grid_search(strategy_func, param_grid, *args, **kwargs):
    """
    Perform a grid search over parameter combinations for a given strategy function.

    Args:
        strategy_func (callable): Strategy function to optimize.
        param_grid (dict): Dictionary of parameter names and lists of values.
        *args: Additional positional arguments for the strategy function.
        **kwargs: Additional keyword arguments for the strategy function.

    Returns:
        dict: Dictionary of results for each parameter combination.
    """
    grid = list(ParameterGrid(param_grid))
    results = {}
    for params in grid:
        key = frozenset(params.items())
        if key in results:
            continue  # Skip duplicate parameter sets
        result = strategy_func(*args, **params, **kwargs)
        results[key] = result
    return results

# Example usage in backtest_strategies.py:
# best_params, best_score, df, model = optimize_event_day_reversion(events, price, thresholds, hold_periods)
# print('Best:', best_params, 'Score:', best_score)
