import numpy as np
from scipy.stats import t, norm, skew, kurtosis
from scipy.optimize import minimize
from scipy.special import gamma
import pandas as pd
import matplotlib.pyplot as plt

def fit_skewt(returns, symbol):
    # Define the negative log-likelihood function for skew-t
    def neg_loglik_skewt(params):
        xi, omega, alpha, nu = params
        # Enforce positivity of scale and df
        if omega <= 0 or nu <= 0:
            return np.inf
        z = (returns - xi) / omega
        # Compute PDF and CDF for each data point
        pdf_vals = t.pdf(z, df=nu) 
        cdf_vals = t.cdf(alpha * z * np.sqrt((nu+1)/(nu + z**2)), df=nu+1)
        # Small value clipping to avoid log(0)
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        cdf_vals = np.clip(cdf_vals, 1e-12, 1-1e-12)
        log_vals = np.log(2/omega) + np.log(pdf_vals) + np.log(cdf_vals)
        return -np.sum(log_vals)  # negative log-likelihood

    def skewt_pdf(x, xi, omega, alpha, nu):
        z = (x - xi) / omega
        pdf_vals = t.pdf(z, df=nu)
        cdf_vals = t.cdf(alpha * z * np.sqrt((nu+1)/(nu + z**2)), df=nu+1)
        return 2/omega * pdf_vals * cdf_vals

    # Initial guess for [xi, omega, alpha, nu]
    init = [0.0, np.std(returns), 0.0, 10.0]  # e.g., zero mean, sample std, no skew, moderate nu
    result = minimize(neg_loglik_skewt, init, 
                     method='BFGS',  # Explicitly specify L-BFGS-B method
                    #  bounds=[(None,None),(1e-6,None),(None,None),(1e-6,None)],
                     options={'maxiter': 1000})  # Optional: increase max iterations if needed
    print(f"\n{symbol} MLE parameters:", result.x)
    xi, omega, alpha, nu = result.x

    # Calculate Skew-T mean
    delta = alpha/np.sqrt(1+alpha**2)
    skewt_mean = xi + omega * delta * np.sqrt(nu/np.pi) * gamma((nu-1)/2) / gamma(nu/2)
    print(f"{symbol} Skew-T distribution mean: {skewt_mean:.6f}")

    # Plot the data and fitted PDF
    x = np.linspace(min(returns), max(returns), 1000)
    pdf_fitted = skewt_pdf(x, xi, omega, alpha, nu)

    # Generate large sample from fitted skew-t
    N = 1000000
    # Generate skew-t by the method: Y = (delta*|U| + sqrt(1-delta^2)*V) / sqrt(W/nu)
    delta = alpha/np.sqrt(1+alpha**2)    # delta = shape in (-1,1)
    U = np.random.normal(size=N)
    V = np.random.normal(size=N)
    W = np.random.chisquare(nu, size=N)
    Z_sn = delta*np.abs(U) + np.sqrt(1 - delta**2)*V  # skew-normal(0,1) variate
    Y = Z_sn / np.sqrt(W/nu)                         # skew-t(0,1,alpha,nu) variate
    sample = xi + omega * Y                          # skew-t(ξ,ω,α,ν) sample

    return x, pdf_fitted, sample, xi, omega, alpha, nu

def fit_studentt(returns, symbol):
    # Define the negative log-likelihood function for Student's t
    def neg_loglik_studentt(params):
        mu, sigma, nu = params
        # Enforce positivity of scale and df
        if sigma <= 0 or nu <= 0:
            return np.inf
        z = (returns - mu) / sigma
        # Compute PDF for each data point
        pdf_vals = t.pdf(z, df=nu)
        # Small value clipping to avoid log(0)
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        log_vals = np.log(1/sigma) + np.log(pdf_vals)
        return -np.sum(log_vals)  # negative log-likelihood

    def studentt_pdf(x, mu, sigma, nu):
        z = (x - mu) / sigma
        return t.pdf(z, df=nu) / sigma

    # Initial guess for [mu, sigma, nu]
    init = [np.mean(returns), np.std(returns), 10.0]  # mean, sample std, moderate nu
    result = minimize(neg_loglik_studentt, init, 
                     method='BFGS',
                     options={'maxiter': 1000})
    print(f"\n{symbol} Student's t MLE parameters:", result.x)
    mu, sigma, nu = result.x

    # Calculate Student's t mean
    studentt_mean = mu  # For nu > 1, the mean is just mu
    print(f"{symbol} Student's t distribution mean: {studentt_mean:.6f}")

    # Plot the data and fitted PDF
    x = np.linspace(min(returns), max(returns), 1000)
    pdf_fitted = studentt_pdf(x, mu, sigma, nu)

    # Generate large sample from fitted Student's t
    N = 1000000
    sample = mu + sigma * np.random.standard_t(nu, size=N)

    return x, pdf_fitted, sample, mu, sigma, nu

# Read the data
df = pd.read_csv('output/daily_returns.csv')

# Process each symbol and collect statistics
stats_list = []
for symbol in df['symbol'].unique():
    symbol_returns = df[df['symbol'] == symbol]['daily_return'].dropna().to_numpy()
    
    # Plot distribution for each stock
    plt.figure(figsize=(10, 6))
    plt.hist(symbol_returns, bins=50, density=True, alpha=0.7, label='Returns')
    
    # Calculate and add Parametric VaR
    parametric_var_95 = np.percentile(symbol_returns, 5)  # 95% VaR (5th percentile)
    # plt.axvline(parametric_var_95, color='orange', linestyle='-.', label=f'95% Parametric VaR: {parametric_var_95:.4f}')
    
    # Calculate and add Parametric CVaR (mean of worst 5%)
    worst_5_percent = symbol_returns[symbol_returns <= parametric_var_95]
    cvar_95 = np.mean(worst_5_percent)
    plt.axvline(cvar_95, color='red', linestyle='--', label=f'95% Parametric CVaR: {cvar_95:.4f}')
    
    # Add normal distribution
    normal_mean = np.mean(symbol_returns)
    normal_std = np.std(symbol_returns)
    x = np.linspace(normal_mean - 4*normal_std, normal_mean + 4*normal_std, 100)
    plt.plot(x, norm.pdf(x, normal_mean, normal_std), 'r-', lw=2, label='Normal Distribution')
    
    # Add vertical line for mean
    # plt.axvline(mean, color='green', linestyle='--', label=f'Mean: {mean:.4f}')
    
    # Add 95% confidence interval
    # ci_lower = mean - 1.96 * std
    # plt.axvline(ci_lower, color='orange', linestyle=':', label=f'95% VaR: [{ci_lower:.4f}]')
    

    #  Calculate and add CVaR for normal distribution
    z_95 = norm.ppf(0.05)
    normal_var_95 = normal_mean - normal_std * (norm.pdf(z_95) / 0.05)
    normal_cvar_95 = normal_mean - normal_std * (norm.pdf(z_95) / 0.05)
    plt.axvline(normal_cvar_95, color='purple', linestyle='--', label=f'95% Normal CVaR: {normal_cvar_95:.4f}')
    
    x, pdf_fitted, sample, xi, omega, alpha, nu = fit_skewt(symbol_returns, symbol)
    plt.plot(x, pdf_fitted, 'b-', lw=2, label='Skew-t Distribution')

    # Estimate 95% VaR and CVaR
    skew_t_var95 = np.percentile(sample, 5)                 # 5th percentile (VaR at 95% conf.)
    skew_t_cvar95 = sample[sample <= skew_t_var95].mean()          # average of returns beyond VaR
    plt.axvline(skew_t_cvar95, color='blue', linestyle='--', label=f'95% Skew-t CVaR: {skew_t_cvar95:.4f}')

    x, pdf_fitted, sample, mu, sigma, nu = fit_studentt(symbol_returns, symbol)
    plt.plot(x, pdf_fitted, 'g-', lw=2, label='Student-t Distribution')

    # Estimate 95% VaR and CVaR
    student_t_var95 = np.percentile(sample, 5)                 # 5th percentile (VaR at 95% conf.)
    student_t_cvar95 = sample[sample <= student_t_var95].mean()          # average of returns beyond VaR
    plt.axvline(student_t_cvar95, color='yellow', linestyle='--', label=f'95% Student-t CVaR: {student_t_cvar95:.4f}')

    plt.title(f'Daily Returns for {symbol}')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'output/{symbol}_returns_distribution.png')
    plt.close()
    
    stats = {
        'symbol': symbol,
        'parametric_var95': round(parametric_var_95 * 100, 2),
        'parametric_cvar95': round(cvar_95 * 100, 2),
        'normal_var95': round(normal_var_95 * 100, 2),
        'normal_cvar95': round(normal_cvar_95 * 100, 2),
        'skew_t_var95': round(skew_t_var95 * 100, 2),
        'skew_t_cvar95': round(skew_t_cvar95 * 100, 2),
        'student_t_var95': round(student_t_var95 * 100, 2),
        'student_t_cvar95': round(student_t_cvar95 * 100, 2),
        'mean': normal_mean,
        'std': normal_std,
        'skew_t_xi': xi,
        'skew_t_omega': omega,
        'skew_t_alpha': alpha,
        'skew_t_nu': nu,
        'student_t_mu': mu,
        'student_t_sigma': sigma,
        'student_t_nu': nu
    }
    stats_list.append(stats)

# Create DataFrame and save to CSV
stats_df = pd.DataFrame(stats_list)
stats_df.to_csv('output/statistics_summary.csv', index=False)
print("\nStatistics summary has been saved to 'output/statistics_summary.csv'")

