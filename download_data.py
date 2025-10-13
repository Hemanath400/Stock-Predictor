import pandas as pd
import yfinance as yf
from datetime import datetime,timedelta

print("Downloading the tesla Stocks")

end_date=datetime.now()
start_date=end_date-timedelta(days=750)

ticker="TSLA"

data=yf.download(ticker,start=start_date,end=end_date)

data.to_csv("TEsla_stock_data.csv")
print(f"Downloaded {len(data)}days of tesla stock")
print(data.head())
print(f"\n DataShape{data.shape}")


