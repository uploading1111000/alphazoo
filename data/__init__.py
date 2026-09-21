import pandas as pd
import yfinance
import requests
RAW_DATA_PATH = 'data/raw_data.csv'

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

try:
    raw_df = pd.read_csv(RAW_DATA_PATH)
except FileNotFoundError:
    raw_df = download()

tickers = raw_df['Ticker'].unique()