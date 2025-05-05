from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd

stock_client = StockHistoricalDataClient(
    api_key="PKVG4YA9NJEB5ZTPH9IV",
    secret_key="N6pUrWJXqS9asyYlvJ089Inpvs7U8UfufkQftkDt",
)

tickers = ["AAPL", "MSFT", "HOOD"]
start_date = "2024-04-30"
end_date = "2025-04-30"
timeframe = TimeFrame.Day

req = StockBarsRequest(
    symbol_or_symbols=tickers,
    timeframe=timeframe,
    start=start_date,
    end=end_date
)

bars_dict = stock_client.get_stock_bars(req)

all_data = []
for ticker in tickers:
    if ticker not in bars_dict.data.keys() or len(bars_dict[ticker]) == 0:
        print(f"No data returned for {ticker}")
        continue
        
    for bar in bars_dict[ticker]:
        all_data.append({
            'symbol': bar.symbol,
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
        })

df = pd.DataFrame(all_data)
df.set_index('timestamp', inplace=True)
df.sort_index(inplace=True)

# Export to CSV
output_file = 'output/stock_data.csv'
df.to_csv(output_file)
print(f"Data exported to {output_file}")
