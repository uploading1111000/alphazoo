import data
import pandas as pd

d = data.get_date_range_lookback(pd.to_datetime("2020-01-01"),pd.to_datetime("2026-01-01"))
for t in d:
    print(len(t['CLOSE']))