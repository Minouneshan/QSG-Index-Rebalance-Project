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
downloaded_tickers = set(price.index.get_level_values('Ticker').unique())
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


# ===== 3. Run Strategies =====

print(">>> Running Post-Announcement Momentum Strategy...")
bt, bt_summary = post_announcement_momentum(events, price, save_fig=True, show_fig=False)
bt.to_csv("results/post_announcement_momentum.csv", index=False)


print(">>> Running Event Day Reversion Strategy...")
rv, rv_summary = event_day_reversion(events, price, save_fig=True,show_fig=False)
rv.to_csv("results/event_day_reversion.csv", index=False)

print(">>> Running Buy-and-Hold Strategy...")
bh, bh_summary = buy_and_hold(events, price, save_fig=True,show_fig=False)
bh.to_csv("results/buy_and_hold.csv", index=False)

print(">>> Running Hedged Momentum (5-day hold) Strategy...")
hm, hm_summary = hedged_momentum(events, price, spy_df, hold_days=5, save_fig=True,show_fig=False)
hm.to_csv("results/hedged_momentum_5d.csv", index=False)

print(">>> Running Holding Period Sweep (1-10 days)...")
sweep_df, sweep_summary, sweep_pnls = holding_period_sweep(
    events, price, spy_df, min_days=1, max_days=10, save_fig=True,show_fig=False
)
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
