import numpy as np
from scipy.stats import t as t_dist

mu_hat = np.array([0.00210133, 0.00643774, 0.00095525])
Sigma_hat = np.array([[1.74239124e-04, 1.52280579e-04, 7.44363741e-05],
                      [1.52280579e-04, 1.19772779e-03, 1.59879186e-04],
                      [7.44363741e-05, 1.59879186e-04, 1.12063719e-04]])
nu_hat = 3.1600
C_constant = -3.165125233164976

# Given a example of portfolio weights, calculate the 1‑day VaR and CVaR
w = np.array([0.35, 0.40, 0.25])
alpha = 0.05

mu_p  = float(w @ mu_hat)
sigma2_p = float(w @ Sigma_hat @ w)
scale_p = np.sqrt(sigma2_p)

# VaR
VaR = t_dist.ppf(alpha, df=nu_hat, loc=mu_p, scale=scale_p)

# CVaR
CVaR = mu_p + scale_p * C_constant

print("\n=== 1‑Day Risk Measures (95% confidence) ===")
print(f"VaR  (5th percentile): {VaR:.4%}")
print(f"CVaR (Expected Shortfall): {CVaR:.4%}")