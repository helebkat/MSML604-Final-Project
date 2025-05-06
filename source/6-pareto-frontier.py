import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt

# Load the MLE parameters from the .npz file
params = np.load("output/mvt_params.npz")

mu_hat = params["mu"]
Sigma_hat = params["Sigma"]
nu_hat = params["nu"].item() 
alpha = 0.05

print("mu_hat:", mu_hat)
print("Sigma_hat:\n", Sigma_hat)
print("nu_hat:", nu_hat)


z = t_dist.ppf(alpha, df=nu_hat)
C_constant = -(t_dist.pdf(z, df=nu_hat) * (nu_hat + z**2) /
               ((nu_hat - 1) * alpha)) # constant C for CVaR from Student-t


def expected_return(w):
    return np.dot(w, mu_hat)

def portfolio_variance(w):
    return np.dot(w, np.dot(Sigma_hat, w))

def portfolio_cvar(w):
    mu_p = expected_return(w)
    sigma_p = np.sqrt(portfolio_variance(w))
    return -(mu_p + sigma_p * C_constant)  # negate for minimization

# optimization
n_assets = len(mu_hat)
target_returns = np.linspace(min(mu_hat), max(mu_hat), 50)

pareto_returns = []
pareto_cvars = []
pareto_weights = []

bounds = [(0, 1)] * n_assets
constraint_sum_to_1 = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

for r_target in target_returns:
    constraints = [
        constraint_sum_to_1,
        {'type': 'ineq', 'fun': lambda w, r=r_target: expected_return(w) - r}
    ]
    
    w0 = np.ones(n_assets) / n_assets
    result = minimize(portfolio_cvar, w0, bounds=bounds, constraints=constraints)
    
    if result.success:
        w_opt = result.x
        pareto_weights.append(w_opt)
        pareto_returns.append(expected_return(w_opt))
        pareto_cvars.append(-portfolio_cvar(w_opt))  # undo negation for actual value


plt.figure(figsize=(10, 6))
plt.plot(pareto_cvars, pareto_returns, marker='o', label='Pareto Frontier')
plt.xlabel("CVaR (Expected Shortfall)")
plt.ylabel("Expected Return")
plt.title("Pareto Frontier: CVaR vs Expected Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
