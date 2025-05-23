# src/backtest_strategies.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_prep import load_event_data, download_price_data, calc_adv

excel_path = "data/Index Add Event Data.xlsx"

# Load & prepare data
events = load_event_data(excel_path)
tickers = sorted(events['Yahoo_Ticker'].unique())
price = download_price_data(tickers)
price = calc_adv(price)

# Filter tickers present in both events and price
downloaded_tickers = set(price.index.get_level_values('Ticker').unique())
events = events[events['Yahoo_Ticker'].isin(downloaded_tickers)]

# Download SPY
spy = download_price_data(['SPY'], start=price.index.get_level_values('Date').min())
spy = spy[['Open', 'Close']].reset_index()
spy.columns = ['Date', 'Ticker', 'Open', 'Close']
spy = spy.pivot(index='Date', columns='Ticker', values=['Open', 'Close'])
spy.columns = ['spy_open', 'spy_close']
spy = spy.reset_index()
spy['Date'] = pd.to_datetime(spy['Date'])

# Merge SPY data
events = events.merge(spy, left_on='Trade Date', right_on='Date', how='left')
events = events.drop(columns=['Date'])

# Now, import and call your strategy modules here (see below)
from strategies import run_all_strategies

if __name__ == "__main__":
    run_all_strategies(events, price, spy)
