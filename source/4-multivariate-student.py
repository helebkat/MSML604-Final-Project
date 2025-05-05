import pandas as pd
import numpy as np
import scipy.optimize as opt
from scipy.special import gammaln
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt

df_long = pd.read_csv("output/daily_returns.csv")
returns_wide = (df_long
                .pivot(index="timestamp", columns="symbol", values="daily_return")
                .dropna())          # shape: (n_days, 3)

X = returns_wide.values                     # numpy array (n, p)
n, p = X.shape

# 2. Negative log‑likelihood for the multivariate Student‑t
def negloglik(params, X):
    n, p = X.shape
    mu = params[:p]
    idx = p
    diag_params = params[idx:idx + p]; idx += p          # unconstrained
    diag_L = np.exp(diag_params)                         # >0
    L = np.zeros((p, p))
    for i in range(p):
        L[i, i] = diag_L[i]
    # lower‑triangular off‑diagonal entries
    for i in range(1, p):
        for j in range(i):
            L[i, j] = params[idx]
            idx += 1
    Sigma = L @ L.T                                      # positive‑definite
    nu = params[-1]

    Xc = X - mu
    inv_Sigma = np.linalg.inv(Sigma)
    quad = np.sum((Xc @ inv_Sigma) * Xc, axis=1)         # Mahalanobis²
    logdet = np.log(np.linalg.det(Sigma))

    term1 = gammaln((nu + p) / 2) - gammaln(nu / 2)
    const = p / 2 * np.log(nu * np.pi) + 0.5 * logdet
    loglik = n * term1 - n * const - (nu + p) / 2 * np.sum(np.log1p(quad / nu))
    return -loglik   

mu0 = X.mean(axis=0)
S0 = np.cov(X, rowvar=False)
L0 = np.linalg.cholesky(S0)
params0 = np.concatenate([
    mu0,                           # μ (p)
    np.log(np.diag(L0)),           # log‑diagonal of L (p)
    L0[np.tril_indices(p, -1)],    # lower off‑diagonals (p(p-1)/2)
    np.array([8.0])        # log(ν‑2), here ν≈10
])

result = opt.minimize(
    negloglik, params0, args=(X,),
    method="BFGS",
    options={"maxiter": 10000, "disp": False}
)

def unpack(params, p):
    mu = params[:p]
    idx = p
    diag_params = params[idx:idx + p]; idx += p
    diag_L = np.exp(diag_params)
    L = np.zeros((p, p))
    for i in range(p):
        L[i, i] = diag_L[i]
    for i in range(1, p):
        for j in range(i):
            L[i, j] = params[idx]
            idx += 1
    Sigma = L @ L.T
    nu = params[-1]
    return mu, Sigma, nu

mu_hat, Sigma_hat, nu_hat = unpack(result.x, p)

print("=== MLE parameters for multivariate Student‑t ===")
print(f"ν (degrees of freedom): {nu_hat:0.4f}")
print(f"μ (mean vector):        {mu_hat}")
print("Σ (scale matrix):")
print(Sigma_hat)

# 6. Portfolio weights and 1‑day VaR / CVaR
w = np.array([0.35, 0.40, 0.25])
mu_p  = float(w @ mu_hat)
sigma2_p = float(w @ Sigma_hat @ w)                      # scale² parameter
scale_p = np.sqrt(sigma2_p)                              # Student‑t scale
alpha = 0.05                                             # 95% confidence (left 5% tail)

# Value‑at‑Risk
VaR = t_dist.ppf(alpha, df=nu_hat, loc=mu_p, scale=scale_p)

# Conditional VaR (Expected Shortfall) – analytic formula
z = t_dist.ppf(alpha, df=nu_hat)                         # standardised quantile
ES_standard = -(t_dist.pdf(z, df=nu_hat) * (nu_hat + z**2) /
                ((nu_hat - 1) * alpha))
CVaR = mu_p + scale_p * ES_standard

print("\n=== 1‑Day Risk Measures (95% confidence) ===")
print(f"VaR  (5th percentile): {VaR:.4%}")
print(f"CVaR (Expected Shortfall): {CVaR:.4%}")