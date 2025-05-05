# MSML604 Final Project

## Proposal

- [Proposal](proposal.md)

## Source Code

### 1. Retrieve Data

[1-retrieve-data.py](source/1-retrieve-data.py)

Generates the CSV file `stock_data.csv` with the daily stock data for the past year. It includes the opening, high, low, closing, and volume of the stock.

### 2. Compute Returns

[2-compute-returns.py](source/2-compute-returns.py)

Reads the CSV file `stock_data.csv` and computes the daily percentage returns of the stock. Adds a new column `daily_return` to the DataFrame. Generates the CSV file `daily_returns.csv`.

### 3. Fit MLE

[3-fit-mle.py](source/3-fit-mle.py)

Reads the CSV file `daily_returns.csv` and fits the MLE for the multivariate normal distribution.

## Next Steps

