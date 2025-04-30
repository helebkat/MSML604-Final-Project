# MSML604 Final Project

## Proposal

- [Proposal](proposal.md)

## Next Steps

- Review the optimization formula and constraints
    - How is $\gamma$ defined?
    - Add an expected return constraint (How do we calculate the expected return?)
    - Give more reasons to why we would like to minimize the CVaR
    - Consider multiple goals optimization formulas
- Review the method to estimate the VaR
    - Should we verify that normal distribution is a good fit for the data?
    - Analyze the possibility of using Monte Carlo simulation
    - Analyze if other distributions can be used as they're a better fit, or if we can add skewness and kurtosis to the normal distribution
    - Analyze "Exponentially Weighted Moving Average (EWMA) VaR Calculation" [1](https://www.mathworks.com/help/risk/estimate-var-using-parametric-methods.html)
- Analyze which algorithms to use for integer optimization
    - Is CPLEX an option?
