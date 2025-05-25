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
downloaded_tickers = set(price.index.get_level_values('Date').unique())
events = events[events['Yahoo_Ticker'].isin(downloaded_tickers)]

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
    bt, bt_summary = post_announcement_momentum(events, price, hold_days=best_params_pa['hold_days'], save_fig=True, show_fig=False)
    bt.to_csv("results/post_announcement_momentum.csv", index=False)
else:
    print("Skipping Post-Announcement Momentum: No optimal parameters found or no trades generated.")

# 2. Optimize and run Event Day Reversion
print(">>> Optimizing Event Day Reversion Parameters...")
from sweeps import sweep_event_day_reversion
param_grid_edr = {'thresholds': [[0.001, 0.002, 0.005, 0.01]], 'hold_periods': [list(range(1, 6))]}
best_params_edr, best_score_edr, _, _ = optimize_with_ml(sweep_event_day_reversion, param_grid_edr, events, price, target_metric='sharpe')
print(f"Best Event Day Reversion params: {best_params_edr}, Sharpe: {best_score_edr}")
if best_params_edr and 'param_0' in best_params_edr and 'param_1' in best_params_edr:
    rv, rv_summary = event_day_reversion(events, price, threshold=best_params_edr.get('param_0', 0.001), hold_days=best_params_edr.get('param_1', 1), save_fig=True, show_fig=False)
    rv.to_csv("results/event_day_reversion.csv", index=False)
else:
    print("Skipping Event Day Reversion: No optimal parameters found or no trades generated.")

# 3. Optimize and run Buy-and-Hold
print(">>> Optimizing Buy-and-Hold Parameters...")
from sweeps import sweep_buy_and_hold
param_grid_bh = {'entry_lags': [[0, 1, 2, 3]], 'hold_periods': [list(range(1, 6))]}
best_params_bh, best_score_bh, _, _ = optimize_with_ml(sweep_buy_and_hold, param_grid_bh, events, price, target_metric='sharpe')
print(f"Best Buy-and-Hold params: {best_params_bh}, Sharpe: {best_score_bh}")
if best_params_bh and 'param_0' in best_params_bh and 'param_1' in best_params_bh:
    bh, bh_summary = buy_and_hold(events, price, entry_lag=best_params_bh.get('param_0', 0), hold_days=best_params_bh.get('param_1', 1), save_fig=True, show_fig=False)
    bh.to_csv("results/buy_and_hold.csv", index=False)
else:
    print("Skipping Buy-and-Hold: No optimal parameters found or no trades generated.")

# 4. Optimize and run Hedged Momentum
print(">>> Optimizing Hedged Momentum Parameters...")
from sweeps import sweep_hedged_momentum
param_grid_hm = {'hedge_ratios': [[0.5, 1.0]], 'hold_periods': [list(range(3, 8))], 'min_advs': [[1, 5, 10]]}
best_params_hm, best_score_hm, _, _ = optimize_with_ml(sweep_hedged_momentum, param_grid_hm, events, price, target_metric='sharpe', spy=spy_df)
print(f"Best Hedged Momentum params: {best_params_hm}, Sharpe: {best_score_hm}")
if best_params_hm and 'param_0' in best_params_hm and 'param_1' in best_params_hm and 'param_2' in best_params_hm:
    hm, hm_summary = hedged_momentum(
        events, price, spy_df,
        hedge_ratio=best_params_hm.get('param_0', 1.0),
        hold_days=best_params_hm.get('param_1', 5),
        min_adv=best_params_hm.get('param_2', 1),
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
