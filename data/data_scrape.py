import pandas as pd
import yfinance
import requests
import numpy as np
import json


RAW_DATA_PATH = 'data/'
QUARTER_PATH = 'data/quarters/'
TICKER_PATH = 'data/tickers/'

def download():
    url = "https://en.wikipedia.org/wiki/FTSE_250_Index"
    headers = {
        "User-Agent": "AlphaZooUK/1.0 (uploading1111000@gmail.com)"
    }
    response = requests.get(url, headers=headers)
    tables = pd.read_html(response.text, match="Ticker")
    tickers = [ticker + '.L' for ticker in tables[0]['Ticker'].tolist()]
    raw_df = yfinance.download(tickers,start='2019-01-01', repair=True)
    stack_df = raw_df.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    stack_df.to_csv(RAW_DATA_PATH + 'raw_data.csv',index=False)
    return stack_df

def split_quarter():
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH  + 'raw_data.csv', parse_dates=[0])
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

names = [f'{quarter}Q{year}' for year in range(2019,2026) for quarter in range(1,5)] + ['1Q2026', '2Q2026']

def get_quarters():
    quarters = {}
    try:
        for name in names:
            quarters[name] = pd.read_csv(QUARTER_PATH + name + '.csv',parse_dates=[0])
    except FileNotFoundError:
        print("Data not split, splitting...")
        quarters = split_quarter()
    return quarters
        
def split_tickers():
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH + 'raw_data.csv', parse_dates=[0])
    except FileNotFoundError:
        print("Data not found, downloading...")
        raw_df = download()

    tickers = raw_df['Ticker'].unique()
    jdict = []
    for ticker in tickers:
        ticker_df = raw_df[raw_df['Ticker']==ticker].copy()
        ticker_df.drop('Ticker',axis=1,inplace=True)
        ticker_df.dropna(axis=0,inplace=True)
        ticker_df.sort_values('Date', inplace=True)
        ticker_df.to_csv(TICKER_PATH + ticker + '.csv',index=False)
        loaded_histories[ticker] = ticker_df
        min_date = ticker_df['Date'].min(skipna=False)
        max_date = ticker_df['Date'].max(skipna=False)
        valid_quarters = set()
        for name in names:
            P = pd.Period(name,freq='Q')
            mask = (ticker_df['Date'] >= P.start_time) & (ticker_df['Date'] < P.end_time)
            rows = ticker_df.loc[mask]
            if len(rows) == 0:
                continue
            if rows.isna().sum().sum() > 0:
                continue
            valid_quarters.add(name)
        jdict.append({
            'name': ticker, 
            'min_date': min_date.strftime('%Y-%m-%d'),
            'max_date': max_date.strftime('%Y-%m-%d'), 
            'file': TICKER_PATH + ticker + '.csv',
            'valid_quarters' : list(valid_quarters)
        })
    with open(TICKER_PATH + 'meta.json', 'w') as f:
        f.write(json.dumps(jdict,indent=4))
    
loaded_histories = {}
def get_all_histories():
    try:
        with open(TICKER_PATH + "meta.json", 'r') as f:
            jdict =  json.load(f)
        for metadata in jdict:
            loaded_histories[metadata['name']] = pd.read_csv(metadata['file'],parse_dates=[0])
    except FileNotFoundError:
        split_tickers()

def trading_days():
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH + 'raw_data.csv', parse_dates=[0])
    except FileNotFoundError:
        print("Data not found, downloading...")
        raw_df = download()
    raw_df.sort_values('Date',inplace=True)
    datetimes = raw_df['Date'].unique().tolist()
    trading_days = [date.strftime('%Y-%m-%d') for date in datetimes]
    with open(RAW_DATA_PATH + 'tradeable_days.json','w') as f:
        json.dump(trading_days, f)
    
    
    

def get_for_quarters():
    pass