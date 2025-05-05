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

#### Skew-t Distribution

Function **neg_loglik_skewt** implements the negative log-likelihood function for the skew-t distribution.

Function **skewt_pdf** implements the probability density function for the skew-t distribution.

Function **fit_skewt** fits the skew-t distribution to the daily returns of the stock. It returns:
- x: the range of source values to plot the PDF
- pdf_fitted: the PDF of the x values
- sample: a large sample from the fitted skew-t distribution
- xi: the location parameter
- omega: the scale parameter
- alpha: the shape parameter
- nu: the degrees of freedom

#### Student's t Distribution

Function **neg_loglik_studentt** implements the negative log-likelihood function for the Student's t distribution.

Function **fit_studentt** fits the Student's t distribution to the daily returns of the stock. It returns:
- x: the range of source values to plot the PDF
- pdf_fitted: the PDF of the x values
- sample: a large sample from the fitted Student's t distribution
- mu: the location parameter
- sigma: the scale parameter
- nu: the degrees of freedom

### 4. Multivariate Student-t Distribution

[4-multivariate-student.py](source/4-multivariate-student.py)

Prints the parameters required to calculate the CVaR of a portfolio:
- Mean
- Sigma
- Nu
- Constant C

### 5. Calculate VaR and CVaR

[5-calculate-var-cvar.py](source/5-calculate-var-cvar.py)

Given a example of portfolio weights, calculate the 1‑day VaR and CVaR
- VaR
- CVaR
## Next Steps

