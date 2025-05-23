# src/data_prep.py
import pandas as pd
import yfinance as yf

def clean_ticker(ticker):
    t = ticker.replace(' US', '').replace(' ', '')
    t = t.replace('.A', '-A').replace('.B', '-B')
    t = t.replace('-W', '-WT').replace('*', '')
    return t

def load_event_data(excel_path):
    events = pd.read_excel(
        excel_path,
        sheet_name='Data',
        parse_dates=['Announced', 'Trade Date']
    )
    events = events[events['Action'].str.contains('Add', na=False)]
    events['first_tradable'] = events['Announced'] + pd.Timedelta(days=1)
    events = events.dropna(subset=['Ticker'])
    events['Yahoo_Ticker'] = events['Ticker'].apply(clean_ticker)
    return events

def download_price_data(tickers, start="2022-05-01", batch_size=50):
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

def calc_adv(price):
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
