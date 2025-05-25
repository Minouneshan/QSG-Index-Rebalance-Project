"""
backtest_strategies.py

This module orchestrates the backtesting of all trading strategies for the QSG Index Rebalance Project.

Functionality:
- Loads data, runs all strategy modules, and saves results for reporting.
- Handles parameter sweeps, summary table generation, and figure saving.
- Designed for reproducibility, clarity, and easy extension to new strategies or workflows.

Readability:
- All functions are self-contained and use clear, descriptive names and docstrings.
- Designed for clarity, extensibility, and easy integration with other modules.

This file is part of a modular pipeline, with supporting strategy code in strategies.py, utilities in utils.py, and data preparation in data_prep.py.
"""

# src/backtest_strategies.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


from data_prep import load_event_data, download_price_data, calc_adv
from strategies import (
    post_announcement_momentum,
    event_day_reversion,
    buy_and_hold,
    hedged_momentum,
    holding_period_sweep
)
from optimizer import optimize_post_announcement, optimize_with_ml

# ===== 0. Output Setup =====
os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# ===== 1. Load & Prepare Data =====
excel_path = "data/Index Add Event Data.xlsx"
events = load_event_data(excel_path)
tickers = sorted(events['Yahoo_Ticker'].unique())
price = download_price_data(tickers)
price = calc_adv(price)

# Only keep events with price data available
# FIX: Use Ticker, not Date, to filter events
if 'Ticker' in price.index.names:
    downloaded_tickers = set(price.index.get_level_values('Ticker').unique())
else:
    # fallback for legacy code, try second level
    downloaded_tickers = set(price.index.get_level_values(1).unique())
events = events[events['Yahoo_Ticker'].isin(downloaded_tickers)]

# Further filter: Only keep events where we have price data for the ticker on the event date
available_pairs = set(price.index)
events = events[events.apply(lambda row: (row['Trade Date'], row['Yahoo_Ticker']) in available_pairs, axis=1)]

# ===== 2. Load SPY =====
spy_df = yf.download(
    'SPY',
    start=price.index.get_level_values('Date').min(),
    end=price.index.get_level_values('Date').max() + pd.Timedelta(days=1),
    auto_adjust=False
)[['Open', 'Close']]

spy_df = spy_df.reset_index()

if isinstance(spy_df.columns, pd.MultiIndex):
    spy_df.columns = [col[0] if col[1] in ('', None) else f"{col[0]}_{col[1]}" for col in spy_df.columns]
spy_df.rename(columns={'Open_SPY': 'spy_open', 'Close_SPY': 'spy_close'}, inplace=True)

#spy_df.rename(columns={'Open': 'spy_open', 'Close': 'spy_close'}, inplace=True)
spy_df['Date'] = pd.to_datetime(spy_df['Date'])

# Merge SPY info with events
events = events.merge(
    spy_df[['Date', 'spy_open', 'spy_close']],
    left_on='Trade Date',
    right_on='Date',
    how='left'
)
events = events.drop(columns=['Date'])


# ===== 3. Run Strategies with Optimized Parameters =====

# 1. Optimize and run Post-Announcement Momentum
print(">>> Optimizing Post-Announcement Momentum Parameters...")
hold_periods = list(range(1, 11))
best_params_pa, best_score_pa, _, _ = optimize_post_announcement(events, price, hold_periods, target_metric='sharpe')
print(f"Best Post-Announcement params: {best_params_pa}, Sharpe: {best_score_pa}")
if best_params_pa and 'hold_days' in best_params_pa:
    hold_days = int(best_params_pa['hold_days'])
    bt, bt_summary = post_announcement_momentum(events, price, hold_days=hold_days, save_fig=True, show_fig=False)
    bt.to_csv("results/post_announcement_momentum.csv", index=False)
else:
    print("Skipping Post-Announcement Momentum: No optimal parameters found or no trades generated.")

# 2. Optimize and run Event Day Reversion
print(">>> Optimizing Event Day Reversion Parameters...")
from sweeps import sweep_event_day_reversion
# Wrapper to adapt scalar params to sweep signature
def sweep_event_day_reversion_scalar(events, price, threshold, hold_days, **kwargs):
    return sweep_event_day_reversion(events, price, thresholds=[threshold], hold_periods=[hold_days], **kwargs)
# FIX: Use scalar parameter grid, not nested lists
param_grid_edr = {'threshold': [0.001, 0.002, 0.005, 0.01], 'hold_days': list(range(1, 6))}
best_params_edr, best_score_edr, _, _ = optimize_with_ml(sweep_event_day_reversion_scalar, param_grid_edr, events, price, target_metric='sharpe')
print(f"Best Event Day Reversion params: {best_params_edr}, Sharpe: {best_score_edr}")
if best_params_edr and 'threshold' in best_params_edr and 'hold_days' in best_params_edr:
    threshold = float(best_params_edr.get('threshold', 0.001))
    hold_days = int(best_params_edr.get('hold_days', 1))
    rv, rv_summary = event_day_reversion(events, price, threshold=threshold, hold_days=hold_days, save_fig=True, show_fig=False)
    rv.to_csv("results/event_day_reversion.csv", index=False)
else:
    print("Skipping Event Day Reversion: No optimal parameters found or no trades generated.")

from sweeps import sweep_buy_and_hold, sweep_hedged_momentum
# Wrapper to adapt scalar params to sweep signature for buy-and-hold
def sweep_buy_and_hold_scalar(events, price, entry_lag, hold_days, **kwargs):
    return sweep_buy_and_hold(events, price, entry_lags=[entry_lag], hold_periods=[hold_days], **kwargs)
# Wrapper to adapt scalar params to sweep signature for hedged momentum
def sweep_hedged_momentum_scalar(events, price, hedge_ratio, hold_days, min_adv, **kwargs):
    return sweep_hedged_momentum(events, price, spy=kwargs.get('spy'), hedge_ratios=[hedge_ratio], hold_periods=[hold_days], min_advs=[min_adv])

# 3. Optimize and run Buy-and-Hold
print(">>> Optimizing Buy-and-Hold Parameters...")
param_grid_bh = {'entry_lag': [0, 1, 2, 3], 'hold_days': list(range(1, 6))}
best_params_bh, best_score_bh, _, _ = optimize_with_ml(sweep_buy_and_hold_scalar, param_grid_bh, events, price, target_metric='sharpe')
print(f"Best Buy-and-Hold params: {best_params_bh}, Sharpe: {best_score_bh}")
if best_params_bh and 'entry_lag' in best_params_bh and 'hold_days' in best_params_bh:
    entry_lag = int(best_params_bh.get('entry_lag', 0))
    hold_days = int(best_params_bh.get('hold_days', 1))
    bh, bh_summary = buy_and_hold(events, price, entry_lag=entry_lag, hold_days=hold_days, save_fig=True, show_fig=False)
    bh.to_csv("results/buy_and_hold.csv", index=False)
else:
    print("Skipping Buy-and-Hold: No optimal parameters found or no trades generated.")

# 4. Optimize and run Hedged Momentum
print(">>> Optimizing Hedged Momentum Parameters...")
param_grid_hm = {'hedge_ratio': [0.5, 1.0], 'hold_days': list(range(3, 8)), 'min_adv': [1, 5, 10]}
best_params_hm, best_score_hm, _, _ = optimize_with_ml(sweep_hedged_momentum_scalar, param_grid_hm, events, price, target_metric='sharpe', spy=spy_df)
print(f"Best Hedged Momentum params: {best_params_hm}, Sharpe: {best_score_hm}")
if best_params_hm and 'hedge_ratio' in best_params_hm and 'hold_days' in best_params_hm and 'min_adv' in best_params_hm:
    hedge_ratio = float(best_params_hm.get('hedge_ratio', 1.0))
    hold_days = int(best_params_hm.get('hold_days', 5))
    min_adv = int(best_params_hm.get('min_adv', 1))
    hm, hm_summary = hedged_momentum(
        events, price, spy_df,
        hedge_ratio=hedge_ratio,
        hold_days=hold_days,
        min_adv=min_adv,
        save_fig=True, show_fig=False)
    hm.to_csv("results/hedged_momentum_optimized.csv", index=False)
else:
    print("Skipping Hedged Momentum: No optimal parameters found or no trades generated.")

# 5. Holding Period Sweep (for reference, not optimized)
print(">>> Running Holding Period Sweep (1-10 days)...")
sweep_df, sweep_summary, sweep_pnls = holding_period_sweep(events, price, spy_df, min_days=1, max_days=10, save_fig=True, show_fig=False)
sweep_df.to_csv("results/holding_period_sweep.csv", index=False)

# ===== 4. Save Figures =====

# Example for saving figures (make sure plt.show() is NOT called before save_fig):
def save_fig(filename):
    plt.savefig(f"results/figures/{filename}", bbox_inches="tight")
    plt.close()

# If your strategy modules already call plt.show(), 
# you might want to refactor so you can call save_fig between plot and show.
# Or: manually save after each run, e.g.
# bt_grouped.plot(...)
# save_fig("cum_pnl_post_announcement.png")

print("\nAll strategies run. Results and figures saved in results/ and results/figures/.")

def run_backtests(
    strategies, events, price, spy=None, save_dir=None
):
    """
    Runs a set of backtest strategies and saves their results.
    
    Parameters:
        strategies (dict): Mapping of strategy names to functions.
        events (pd.DataFrame): Event data for the strategies.
        price (pd.DataFrame): Price data for the strategies.
        spy (pd.DataFrame or None): SPY data if required by the strategies.
        save_dir (str or None): Directory to save results. If None, does not save.
    
    Returns:
        dict: Mapping from strategy names to (results, summary) tuples.
    """
    # ...existing code...

def run_all_strategies(event_data, price_data, spy_data):
    """
    Run all trading strategies and save results for reporting.
    Args:
        event_data (pd.DataFrame): Event data for index additions.
        price_data (pd.DataFrame): Price data for all tickers.
        spy_data (pd.DataFrame): Benchmark (SPY) data.
    Returns:
        dict: Dictionary of results for each strategy.
    """
    # ...existing code...

def run_parameter_sweeps(event_data, price_data, spy_data):
    """
    Run parameter sweeps (e.g., holding period, thresholds) for strategies and save results.
    Args:
        event_data (pd.DataFrame): Event data for index additions.
        price_data (pd.DataFrame): Price data for all tickers.
        spy_data (pd.DataFrame): Benchmark (SPY) data.
    Returns:
        dict: Dictionary of sweep results for each strategy.
    """
    # ...existing code...

def generate_summary_table(results):
    """
    Generate a summary table of key metrics for all strategies.
    Args:
        results (dict): Dictionary of strategy results.
    Returns:
        pd.DataFrame: Summary table of metrics.
    """
    # ...existing code...
