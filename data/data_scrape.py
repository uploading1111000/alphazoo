import pandas as pd
import yfinance
import requests
import numpy as np
import json
from io import StringIO
from pathlib import Path


RAW_DATA_PATH = 'data/'
QUARTER_PATH = 'data/quarters/'
TICKER_PATH = 'data/tickers/'

def download():
    url = "https://en.wikipedia.org/wiki/FTSE_250_Index"
    headers = {
        "User-Agent": "AlphaZooUK/1.0 (uploading1111000@gmail.com)"
    }
    response = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(response.text), match="Ticker")
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
    Path(TICKER_PATH).mkdir(parents=True, exist_ok=True)
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
    if lookback < 0 or forward < 0:
        raise ValueError("lookback and forward must be non-negative")

    calendar = pd.DatetimeIndex(trading_days()).normalize().unique().sort_values()
    begin = pd.Timestamp(begin).normalize()
    end = pd.Timestamp(end).normalize()
    if begin >= end:
        raise ValueError("begin must be earlier than end")

    evaluation_dates = calendar[(calendar >= begin) & (calendar < end)]
    if evaluation_dates.empty:
        raise ValueError("The requested interval contains no trading days")

    first_eval = calendar.get_indexer([evaluation_dates[0]])[0]
    last_eval = calendar.get_indexer([evaluation_dates[-1]])[0]
    first_required = first_eval - lookback - forward
    end_required = last_eval + 1
    if first_required < 0 or end_required > len(calendar):
        raise ValueError(
            "The trading calendar does not include the full lookback and forward window"
        )

    # Keep forward-return outcomes within [begin, end); acquire their signal history before begin.
    # One shared, newest-first index keeps every ticker's array positions aligned.
    required_dates = calendar[first_required:end_required][::-1]
    required_fields = ["Open", "High", "Low", "Close", "Volume"]

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
        ticker_df = pd.read_csv(meta['file'], parse_dates=[0])
        if not set(required_fields).issubset(ticker_df.columns):
            continue
        if ticker_df["Date"].duplicated().any():
            continue

        ticker_df["Date"] = pd.to_datetime(ticker_df["Date"]).dt.normalize()
        aligned = ticker_df.set_index("Date")[required_fields].reindex(required_dates)
        aligned = aligned.apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(aligned.to_numpy(dtype=np.float64)).all():
            continue
        if np.any(np.equal(aligned['Volume'],np.zeros(len(aligned)))):
            continue

        # Existing consumers expect one array per field, ordered newest first.
        ticker_data = {
            "close": aligned["Close"].to_numpy(dtype=np.float32),
            "open": aligned["Open"].to_numpy(dtype=np.float32),
            "high": aligned["High"].to_numpy(dtype=np.float32),
            "low": aligned["Low"].to_numpy(dtype=np.float32),
            "volume": aligned["Volume"].to_numpy(dtype=np.float32)
        }
        period_data_list.append(ticker_data)
    if not period_data_list:
        raise ValueError(
            "No ticker has complete finite OHLCV data for the requested window"
        )
    return period_data_list

def initialize_prediction_data(instance, universe, forward, lookback):
    """Populate the shared universe, target, and history data used by predictors."""
    instance.universe_n = len(universe)
    instance.day_length = len(universe[0]["close"])
    instance.targets_n = instance.day_length - forward - lookback
    instance.lookback_n = instance.day_length - forward
    instance.targets = np.zeros((instance.universe_n, instance.targets_n))
    instance.targets[0] = (
        universe[0]["close"][:instance.targets_n]
        / universe[0]["close"][forward:forward + instance.targets_n]
        - 1
    )
    for i, history in enumerate(universe[1:], start=1):
        assert len(history["close"]) == instance.day_length
        instance.targets[i] = (
            history["close"][:instance.targets_n]
            / history["close"][forward:forward + instance.targets_n]
            - 1
        )

    instance.histories = [
        {key: values[forward:] for key, values in history.items()}
        for history in universe
    ]
    print(f"{instance.targets_n} days loaded against {instance.universe_n} tickers!")

AVAILABLE_FIELDS = ["close","open","high","low","volume"]
AVAILABLE_FIELDS_STRING = """close - The stock price at the end of the trading day
open - The stock price at the beginning of the trading day
high - The maximum stock price over the trading day
low - The minimum stock price over the trading day
volume - The volume of the stock traded over the trading day"""

    
