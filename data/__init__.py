import pandas as pd
import yfinance
import requests
import numpy as np
RAW_DATA_PATH = 'data/raw_data.csv'
QUARTER_PATH = 'data/quarters/'

def download():
    url = "https://en.wikipedia.org/wiki/FTSE_250_Index"
    headers = {
        "User-Agent": "AlphaZooUK/1.0 (uploading1111000@gmail.com)"
    }
    response = requests.get(url, headers=headers)
    tables = pd.read_html(response.text, match="Ticker")
    tickers = [ticker + '.L' for ticker in tables[0]['Ticker'].tolist()]
    raw_df = yfinance.download(tickers,start='2020-01-01')
    raw_df.to_csv(RAW_DATA_PATH)
    stack_df = raw_df.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    stack_df.to_csv(RAW_DATA_PATH,index=False)
    return stack_df

def split_quarter():
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH, parse_dates=[0])
    except FileNotFoundError:
        print("Data not found, downloading...")
        raw_df = download()

    tickers = raw_df['Ticker'].unique()
    quarters = {}
    for name in names:
        period = pd.Period(name,freq='Q')
        mask = (raw_df['Date'] >= period.start_time) & (raw_df['Date'] < period.end_time)
        quarters[name] = raw_df.loc[mask]
    for name in quarters:
        for ticker in tickers:
            if quarters[name][quarters[name]['Ticker'] == ticker].isna().sum(axis=1).sum() > 0:
                quarters[name] = quarters[name][quarters[name]['Ticker'] != ticker]
        quarters[name].to_csv(QUARTER_PATH + name + '.csv',index=False)
    return quarters

names = [f'{quarter}Q{year}' for year in range(2020,2026) for quarter in range(1,5)] + ['1Q2026', '2Q2026']

def get_quarters():
    quarters = {}
    try:
        for name in names:
            quarters[name] = pd.read_csv(QUARTER_PATH + name + '.csv',parse_dates=[0])
    except FileNotFoundError:
        print("Data not split, splitting...")
        quarters = split_quarter()
    return quarters

def test():
    quarters = get_quarters()
    n_dates = []
    for name in quarters:
        n_dates.append(len(quarters[name]['Date'].unique()))
    print(min(n_dates))
    print(np.quantile(n_dates,0.25))
    print(np.quantile(n_dates,0.75))
    print(max(n_dates))
        
    