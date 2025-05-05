import pandas as pd
import numpy as np
import scipy.optimize as opt
from scipy.special import gammaln
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt

# Read the daily returns dataset
df_long = pd.read_csv("output/daily_returns.csv")
# Generate a dataframe with the stocks as columns and the dates as rows
returns_wide = (df_long
                .pivot(index="timestamp", columns="symbol", values="daily_return")
                .dropna())

X = returns_wide.values
n, p = X.shape

# Negative log‑likelihood of the multivariate Student‑t
def negloglik(params, X):
    n, p = X.shape
    mu = params[:p]
    # Use the Cholesky decomposition to ensure that on each iteration a positive‑definite matrix is obtained
    idx = p
    diag_params = params[idx:idx + p]; idx += p
    diag_L = np.exp(diag_params)
    L = np.zeros((p, p))
    for i in range(p):
        L[i, i] = diag_L[i]
    # Lower‑triangular off‑diagonal entries
    for i in range(1, p):
        for j in range(i):
            L[i, j] = params[idx]
            idx += 1
    # The regenerated covariance matrix is positive‑definite
    Sigma = L @ L.T
    nu = params[-1]

    # Terms of the log multivariate Student‑t distribution
    Xc = X - mu
    inv_Sigma = np.linalg.inv(Sigma)
    quad = np.sum((Xc @ inv_Sigma) * Xc, axis=1)
    logdet = np.log(np.linalg.det(Sigma))

    term1 = gammaln((nu + p) / 2) - gammaln(nu / 2)
    const = p / 2 * np.log(nu * np.pi) + 0.5 * logdet
    loglik = n * term1 - n * const - (nu + p) / 2 * np.sum(np.log1p(quad / nu))
    return -loglik   

mu0 = X.mean(axis=0)
S0 = np.cov(X, rowvar=False)
L0 = np.linalg.cholesky(S0)
params0 = np.concatenate([
    mu0, # mu
    np.log(np.diag(L0)), # log‑diagonal of L (p)
    L0[np.tril_indices(p, -1)], # lower off‑diagonals (p(p-1)/2)
    np.array([8.0]) # nu
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
print(f"nu (degrees of freedom): {nu_hat:0.4f}")
print(f"mu (mean vector):        {mu_hat}")
print("Sigma (scale matrix):")
print(Sigma_hat)

alpha = 0.05
z = t_dist.ppf(alpha, df=nu_hat)
ES_standard = -(t_dist.pdf(z, df=nu_hat) * (nu_hat + z**2) /
                ((nu_hat - 1) * alpha))
print(f"Constant C: {ES_standard}")

