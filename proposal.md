# MSML 604 Project - Mid-semester Report

# Team Members

- Govind Singhal 

- Lohit Gopikonda

- Damian Calabresi

- Helen Li

- Amaan Mohammed

# Portfolio Risk Minimization based on Conditional Value at Risk (CVaR)

## Introduction

In the context of financial market investment, risk management is one of the main areas of study.

Risk managers conduct assessments to understand the risk-exposure of a given portfolio that combines a series of financial assets (Equities, bonds, etc). This risk understanding is specially important for investment banks that manage their customer portfolios, as they need to ensure they posses the required coverage in case of an unexpected loss.

Value at Risk is a model to understand the expected maximum loss under a normal scenario, or at least with a level of confidence.

Conditional Value at Risk is a measure to predict the expected loss in case that this unexpected events occurs and the Value at Risk threshold is surpassed.

The purpose of this optimization problem will be to find a formula that minimizes the Conditional Value at Risk of an investment portfolio given a series of financial assets availability.

This optimization will rely on historical data to estimate the expected gains and losses from each investment.

## Background

### Portfolio Management

Portfolio management is the process of selecting and overseeing a collection of financial assets in order to achieve specific investment goals while balancing risk and return.

The portfolio management process can focus on different aspects of the investment: liquidity, turnover, volatility, performance benchmark, and risk tolerance.

For the purpose of this optimization problem we are focused on risk tolerance and how to minimize this.

#### Risk Management

Risk management is the practice of identifying, analyzing, and mitigating potential financial losses to maintain portfolio stability and meet investment objectives.

### Value At Risk

One of the key metrics for risk management is Value At Risk. This metric estimates the maximum potential loss of an investment or portfolio over a given time period for a given confidence level.

**Historical calculation**

We want to calculate the **1-day VaR with 95% confidence** for an investment on Apple stocks using historical returns data. We have 10 days of data:

| Day 1  | Day 2  | Day 3  | Day 4  | Day 5  | Day 6  | Day 7  | Day 8  | Day 9  | Day 10 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| 0.053  | -1.234 | 0.789  | -0.456 | 1.234  | -2.567 | 0.345  | 1.678  | -1.890 | 0.123   |

We need to sort the data in ascending order and find the 5th percentile:

| Day 1  | Day 2  | Day 3  | Day 4  | Day 5  | Day 6  | Day 7  | Day 8  | Day 9  | Day 10 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| -2.567 | -1.890 | -1.234 | -0.456 | 0.053  | 0.123  | 0.345  | 0.789  | 1.234  | 1.678   |

The 5th percentile is -1.890. If we have an investment of $100,000, the VaR is $1,890.

This represents the maximum loss we would expect with 95% confidence.

In the case of multiple assets, the measurement of the covariance isn't required as it's already implied in the historical returns data.

**Parametric calculation**

We can also assume the returns of the assets follow a normal distribution and calculate the VaR using the mean and standard deviation of the returns.
$$
VaR = \mu - z_\alpha \cdot \sigma
$$

**Note:** For the median and deviation we can use the daily return in absolute values or percentages.

$z_\alpha$ is the z-score for the given confidence level. For a 95% confidence level, $z_\alpha = 1.645$.

In the case of multiple assets, the parametric calculation makes use of the covariance matrix and the weights of the assets in the portfolio:

$$
Portfolio VaR = w^T \cdot \mu - z_\alpha \cdot \sqrt{w^T \cdot \Sigma \cdot w}
$$

Where $w$ is the vector of weights of the assets in the portfolio and $\Sigma$ is the covariance matrix of the returns.

### Conditional Value at Risk or Expected Shortfall

Conditional Value at Risk (CVaR) is a measure of the expected loss in case that this unexpected events occurs and the Value At Risk threshold is surpassed.

The formula that describes the CVaR is:

$$
CVaR_\alpha = E[Loss | Loss > VaR_\alpha]
$$

Being $f(x)$ the probability density function of the asset returns. The formula to calculate the CVaR of one asset is:

$$
CVaR = \frac{1}{\alpha} \int_{-\infty}^{VaR} x f(x) dx
$$

Similarly to what we have done before, the CVaR can be calculated from historical data or using the parametric approach.

**Note:** $\alpha$ is the confidence level. E.g.: $\alpha = 0.05$ means a 95% confidence level.

**Historical calculation**

- We have $n$ observations sorted in ascending order. We're only going to consider the first $\alpha \cdot n$ observations as the rest are above the VaR threshold.
- If we want to estimate the CVaR with 95% confidence we'll define $\alpha = 0.05$.
- $L_i$ is the loss of the $i$-th observation.

Then CVaR is:

$$
CVaR_\alpha = \frac{1}{\alpha \cdot n} \sum_{i=1}^{\alpha \cdot n} L_i
$$

This is basically the average of the losses of the observations that are below the VaR threshold.

**Parametric calculation**

The parametric calculation is similar to the one we did for the VaR. We need to estimate the mean and standard deviation of the returns. Then we need to get the probability density function of the normal distribution at the z-score of the VaR threshold.

Then cVaR is calculated as:

$$
CVaR_\alpha = \mu + \frac{\phi(z_\alpha)}{\alpha} \cdot \sigma
$$

Similarly to what we've done before, calculating the CVar for multiple assets requites to calculate the standard deviation using the covariance matrix:

$$
CVaR_\alpha = w^T \cdot \mu + \frac{\phi(z_\alpha)}{\alpha} \cdot \sqrt{w^T \cdot \Sigma \cdot w}
$$

Where $w$ is the vector of weights of the assets in the portfolio and $\Sigma$ is the covariance matrix of the returns.

# Optimization Problem

Assuming that we have a set of assets with historical returns data, we can calculate the VaR and CVaR for each asset. We want to find, for a given set of assets, the portfolio weight that minimizes the CVaR of the portfolio.

First, we need to identify a set of variables:

- $n$: number of assets
- $T$: number of days of historical data
- $x$: portfolio weights
- $R$: matrix of historical returns (T days, n assets)
- $\alpha$: confidence level
- $\gamma$: VaR threshold (It will be one of the optimization variables)

**Optimization formula**

$\gamma + \frac{1}{\alpha \cdot T} \sum_{t=1}^{T} max(0, -R_t x - \gamma)$

$max(0, -R_t x - \gamma)$ means that only counts if the VaR threshold is exceeded.

**Constraints:**

- $\sum_{i=1}^{n} x_i = 1$ (All the money will be invested)
- $x_i \geq 0$ (Short-selling not allowed)
- $\gamma \geq 0$ (VaR threshold must be positive)

#### References

- Portfolio Management – Investopedia (https://www.investopedia.com/terms/p/portfoliomanagement.asp)
- Risk Management – Investopedia (https://www.investopedia.com/terms/r/riskmanagement.asp)
- Value at Risk (VaR) – Investopedia (https://www.investopedia.com/terms/v/var.asp)
- Conditional Value at Risk (CVaR) – Investopedia (https://www.investopedia.com/terms/c/conditional_value_at_risk.asp)
- Expected Shortfall – Wikipedia (https://en.wikipedia.org/wiki/Expected_shortfall)
- Andersson, F., Mausser, H., Rosen, D. et al. Credit risk optimization with Conditional Value-at-Risk criterion. Math. Program. 89, 273–291 (2001). https://doi.org/10.1007/PL00011399

