import data
print("Entering")
D = data.get_date_range_lookback("2020-01-01","2021-01-01")
for i in range(5):
    print(D[i]["close"][:5])