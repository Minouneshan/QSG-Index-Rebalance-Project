"""
data_prep.py

This module provides data loading, cleaning, and preparation utilities for the QSG Index Rebalance Project.

Functionality:
- Loads and preprocesses event, price, and benchmark data for use in strategy modules.
- Handles missing data, type conversions, and feature engineering (e.g., ADV calculation).
- Ensures all data is formatted and indexed for robust downstream analysis.

Readability:
- All functions are self-contained and use clear, descriptive names and docstrings.
- Designed for clarity, reproducibility, and easy integration with other modules.

This file is part of a modular pipeline, with supporting strategy code in strategies.py and orchestration in backtest_strategies.py.
"""

# src/data_prep.py
import pandas as pd
import yfinance as yf

def clean_ticker(ticker):
    """
    Cleans the ticker symbol by removing unwanted characters and formatting it for Yahoo Finance.

    Parameters:
        ticker (str): The original ticker symbol.

    Returns:
        str: The cleaned and formatted ticker symbol.
    """
    t = ticker.replace(' US', '').replace(' ', '')
    t = t.replace('.A', '-A').replace('.B', '-B')
    t = t.replace('-W', '-WT').replace('*', '')
    return t

def load_event_data(filepath):
    """
    Load and preprocess index event data from an Excel or CSV file.
    
    Args:
        filepath (str): Path to the event data file.
    
    Returns:
        pd.DataFrame: Cleaned and formatted event data.
    """
    events = pd.read_excel(
        filepath,
        sheet_name='Data',
        parse_dates=['Announced', 'Trade Date']
    )
    events = events[events['Action'].str.contains('Add', na=False)]
    events['first_tradable'] = events['Announced'] + pd.Timedelta(days=1)
    events = events.dropna(subset=['Ticker'])
    events['Yahoo_Ticker'] = events['Ticker'].apply(clean_ticker)
    return events

def download_price_data(tickers, start="2022-05-01", batch_size=50):
    """
    Downloads historical price data from Yahoo Finance for a list of tickers.

    Parameters:
        tickers (list): List of ticker symbols to download data for.
        start (str): Start date for downloading data (format: 'YYYY-MM-DD').
        batch_size (int): Number of tickers to download in each batch (to avoid hitting API limits).

    Returns:
        pd.DataFrame: Historical price data indexed by (date, ticker).
    """
    dfs = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        df = yf.download(batch, start=start, group_by='ticker', threads=True, auto_adjust=False)
        if df.empty: continue
        if len(batch) == 1:
            df.columns = pd.MultiIndex.from_product([df.columns, batch])
        dfs.append(df)
    if not dfs:
        raise ValueError("No data downloaded!")
    price = pd.concat(dfs, axis=1)
    price.columns = price.columns.swaplevel(0, 1)
    price = price.sort_index(axis=1)
    price = price.stack(level=1)
    price.index.names = ['Date', 'Ticker']
    return price

def load_price_data(filepath):
    """
    Load and preprocess price data from a CSV file.
    
    Args:
        filepath (str): Path to the price data file.
    
    Returns:
        pd.DataFrame: Cleaned and formatted price data indexed by (date, ticker).
    """
    price = pd.read_csv(
        filepath,
        parse_dates=['Date'],
        dtype={'Ticker': str}
    )
    price = price.dropna(subset=['Close'])
    price['Ticker'] = price['Ticker'].apply(clean_ticker)
    price = price.set_index(['Date', 'Ticker'])
    return price

def calc_adv(price):
    """
    Calculates the 20-day Average Daily Volume (ADV20) for the given price data.

    Parameters:
        price (pd.DataFrame): Price data containing a 'Volume' column.

    Returns:
        pd.DataFrame: Price data with an additional 'ADV20' column.
    """
    if 'Volume' in price.columns:
        adv20 = (
            price['Volume']
            .unstack('Ticker')
            .rolling(20, min_periods=1)
            .mean()
            .stack()
            .rename('ADV20')
        )
        price = price.join(adv20)
    return price

def calculate_adv(df, window=20):
    """
    Calculate average daily volume (ADV) for each ticker over a rolling window.
    
    Args:
        df (pd.DataFrame): Price data with 'Volume' column.
        window (int): Rolling window size for ADV calculation.
    
    Returns:
        pd.DataFrame: DataFrame with new 'ADV20' column.
    """
    if 'Volume' in df.columns:
        adv = (
            df['Volume']
            .unstack('Ticker')
            .rolling(window, min_periods=1)
            .mean()
            .stack()
            .rename('ADV20')
        )
        df = df.join(adv)
    return df

def merge_spy_data(events, spy):
    """
    Merges SPY price data into the event DataFrame for use in strategies.
    
    Parameters:
        events (pd.DataFrame): Event data.
        spy (pd.DataFrame): SPY price data with columns Date, spy_open, spy_close.
    
    Returns:
        pd.DataFrame: Event data with SPY columns merged in.
    """
    # ...existing code...

def merge_benchmark_data(price_df, benchmark_df):
    """
    Merge benchmark (e.g., SPY) data into price data for event-relative calculations.
    
    Args:
        price_df (pd.DataFrame): Main price data.
        benchmark_df (pd.DataFrame): Benchmark data (e.g., SPY).
    
    Returns:
        pd.DataFrame: Merged DataFrame with benchmark columns added.
    """
    price_df = price_df.join(
        benchmark_df[['spy_close', 'spy_open']],
        on=['Date'],
        how='left'
    )
    return price_df

def preprocess_event_data(events):
    """
    Cleans and preprocesses the event data for downstream analysis.
    
    Parameters:
        events (pd.DataFrame): Raw event data.
    
    Returns:
        pd.DataFrame: Cleaned and preprocessed event data.
    """
    # ...existing code...
