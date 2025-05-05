import pandas as pd

df = pd.read_csv('output/stock_data.csv', parse_dates=['timestamp'])

# Compute daily percentage returns for all symbols
df['daily_return'] = df.groupby('symbol')['close'].pct_change()

# Drop rows with NaN values (first day for each symbol)
df = df.dropna(subset=['daily_return'])

# Write the CSV with returns for all symbols
df.to_csv('output/daily_returns.csv', index=False)
