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
        with open(RAW_DATA_PATH + 'tradeable_days.json','r') as f:
            trading_days_s = json.load(f)
        return [pd.to_datetime(s) for s in trading_days_s]
    except FileNotFoundError:
        print("Tradeable days not found.")
    try:
        raw_df = pd.read_csv(RAW_DATA_PATH + 'raw_data.csv', parse_dates=[0])
    except FileNotFoundError:
        print("Data not found, downloading...")
        raw_df = download()
    raw_df.sort_values('Date',inplace=True)
    datetimes = raw_df['Date'].unique().tolist()
    trading_days_s = [date.strftime('%Y-%m-%d') for date in datetimes]
    with open(RAW_DATA_PATH + 'tradeable_days.json','w') as f:
        json.dump(trading_days_s, f)
    return datetimes

def get_date_range_lookback(begin, end, lookback=90, forward=20):
    try:
        with open(TICKER_PATH + "meta.json",'r') as f:
            ticker_meta = json.load(f)
    except FileNotFoundError:
        print("Ticker metadata not found, splitting...")
        split_tickers()
        with open(TICKER_PATH + "meta.json",'r') as f:
            ticker_meta = json.load(f)
    period_data_list = []
    for meta in ticker_meta:
        min_date = pd.to_datetime(meta["min_date"])
        max_date = pd.to_datetime(meta["max_date"])
        if min_date > begin or max_date <= end:
            continue
        ticker_df = pd.read_csv(meta['file'],parse_dates=[0])
        ticker_df.sort_values('Date',ascending=False,inplace=True)

        for i,date in enumerate(ticker_df['Date']):
            if date < end:
                first_index = i
                break
        else:
            continue
        for i,date in enumerate(ticker_df['Date']):
            if date < begin:
                second_index = i-1
                break
        else:
            continue
        third_index = second_index + lookback + forward
        if third_index > ticker_df.shape[0]:
            continue
        ticker_period_df = ticker_df.iloc[first_index:third_index]
        ticker_data = {
            "close": ticker_period_df['Close'].to_numpy(dtype=np.float32),
            "open": ticker_period_df['Open'].to_numpy(dtype=np.float32),
            "high": ticker_period_df['High'].to_numpy(dtype=np.float32),
            "low": ticker_period_df['Low'].to_numpy(dtype=np.float32),
            "volume": ticker_period_df['Volume'].to_numpy(dtype=np.float32)
        }
        period_data_list.append(ticker_data)
    return period_data_list

AVAILABLE_FIELDS_STRING = """close - The stock price at the end of the trading day
open - The stock price at the beginning of the trading day
high - The maximum stock price over the trading day
low - The minimum stock price over the trading day
volume - The volume of the stock traded over the trading day"""

    
