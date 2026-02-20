import numpy as np
import cvxpy as cp
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.linalg import sqrtm

# --- 1. Robust Parameters Estimation ---
def estimate_robust_params(returns, alpha=0.05):
    N = len(returns)     # Sample size
    k = returns.shape[1] # Number of assets
    
    # Mean Uncertainty (Gamma1) & Covariance Uncertainty (Gamma2)
    gamma1 = chi2.ppf(1 - alpha, df=k) / N
    gamma2 = (N - 1) / (N - k - 1) 
    
    return round(gamma1, 6), round(gamma2, 6)

# --- 2. Get Bounds for Epsilon (Risk Limits) ---
def get_risk_bounds(mu, Sigma):
    n = len(mu)
    w = cp.Variable(n, nonneg=True)
    
    # Minimum Variance
    prob_min = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), [cp.sum(w) == 1])
    prob_min.solve(solver=cp.SCS) 
    min_risk = prob_min.value
    
    # Variance of Max Return Portfolio
    prob_max_ret = cp.Problem(cp.Maximize(mu @ w), [cp.sum(w) == 1])
    prob_max_ret.solve(solver=cp.SCS)
    max_ret_weights = w.value
    max_risk = max_ret_weights.T @ Sigma @ max_ret_weights
    
    return min_risk, max_risk

# --- 3. Strict SDP DRPO Solver (Appendix C.4) ---
def solve_drpo_epigraph(mu_hat, Sigma_hat, gamma1, gamma2, epsilon, tau=0.001):
    n = len(mu_hat)
    
    # Decision Variables
    x = cp.Variable(n, nonneg=True)
    Q = cp.Variable((n, n), PSD=True)
    q = cp.Variable(n)
    r = cp.Variable()
    P = cp.Variable((n, n), symmetric=True)
    p = cp.Variable(n)
    s = cp.Variable(nonneg=True)
    
    # Auxiliary Epigraph Variables
    y = cp.Variable() # Bounding variable for portfolio risk
    v = cp.Variable() # Epigraph variable for the inverse barrier
    
    # Linear Objective (Eq 33)
    objective = cp.Minimize(gamma2 * cp.trace(Sigma_hat @ Q) - 
                            cp.trace(np.outer(mu_hat, mu_hat) @ Q) + 
                            cp.trace(Sigma_hat @ P) - 2 * mu_hat @ p + 
                            gamma1 * s + r + v)
    
    constraints = [
        cp.sum(x) == 1, 
        p == -0.5 * q - Q @ mu_hat
    ]
    
    # Dual Moment LMIs
    lmi_1 = cp.bmat([[P, cp.reshape(p, (n, 1))], 
                     [cp.reshape(p, (1, n)), cp.reshape(s, (1, 1))]])
    constraints.append(lmi_1 >> 0)

    off_diag = cp.reshape(0.5 * q + 0.5 * x, (n, 1))
    lmi_2 = cp.bmat([[Q, off_diag], 
                     [off_diag.T, cp.reshape(r, (1, 1))]])
    constraints.append(lmi_2 >> 0)

    # Risk Bound LMI (Eq 31)
    Sigma_half = sqrtm(Sigma_hat).real
    I = np.eye(n)
    vec_term = Sigma_half @ x
    lmi_risk = cp.bmat([
        [cp.reshape(y, (1, 1)), cp.reshape(vec_term, (1, n))],
        [cp.reshape(vec_term, (n, 1)), I]
    ])
    constraints.append(lmi_risk >> 0)

    # Barrier LMI (Eq 32)
    sqrt_tau = np.sqrt(tau)
    tau_matrix = np.array([[sqrt_tau]])
    lmi_barrier = cp.bmat([
        [cp.reshape(v, (1, 1)), tau_matrix],
        [tau_matrix, cp.reshape(epsilon - y, (1, 1))]
    ])
    constraints.append(lmi_barrier >> 0)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, max_iters=5000, verbose=False) 
        return x.value if prob.status in ["optimal", "optimal_inaccurate"] else None
    except:
        return None

# --- 4. Random Portfolio Generator ---
def generate_portfolio_cloud(mu, sigma, num_samples=8000):
    n = len(mu)
    all_rets, all_risks = [], []
    for _ in range(num_samples):
        w = np.random.dirichlet(np.ones(n), size=1)[0]
        all_rets.append(np.dot(w, mu))
        all_risks.append(np.sqrt(np.dot(w.T, np.dot(sigma, w))))
    return all_risks, all_rets

# --- 5. Main Execution & Visualization ---
if __name__ == "__main__":
    tickers = ['NVDA', 'AMD', 'META', 'AVGO', 'ORCL', 'NFLX', 'ADBE', 'CRM', 'QCOM', 'INTC']
    print(f"Fetching Data for {tickers}...")
    
    data = yf.download(tickers, start="2020-01-01", end="2023-12-31", auto_adjust=True)['Close']
    returns = data.pct_change().dropna()
    mu = returns.mean().values * 252
    sigma = returns.cov().values * 252

    print("Generating Feasible Objective Space Cloud...")
    cloud_risk, cloud_ret = generate_portfolio_cloud(mu, sigma)
    
    print("Calculating Automatic Risk Bounds...")
    risk_min, risk_max = get_risk_bounds(mu, sigma)
    gamma1, gamma2 = estimate_robust_params(returns)
    
    print(f"Data-driven Parameters: Gamma1 = {gamma1}, Gamma2 = {gamma2}")
    print(f"Detected Range: Min Risk = {risk_min:.4f}, Max Risk = {risk_max:.4f}")
    
    # We define epsilon points across the feasible risk range
    epsilon_values = np.linspace(risk_min * 1.05, risk_max * 0.8, 20)
    frontier_risk, frontier_ret = [], []
    
    print("Computing Strict Robust Pareto Front (Epigraph DRPO)...")
    for eps in epsilon_values:
        w = solve_drpo_epigraph(mu, sigma, gamma1, gamma2, epsilon=eps, tau=0.001) 
        if w is not None:
            frontier_risk.append(np.sqrt(w.T @ sigma @ w))
            frontier_ret.append(w.T @ mu)

    # --- Plotting ---
    plt.figure(figsize=(10, 7))
    plt.scatter(cloud_risk, cloud_ret, c='royalblue', s=10, alpha=0.4, label='Feasible Objective Space')
    plt.plot(frontier_risk, frontier_ret, color='red', marker='o', markersize=4, linewidth=2, label='Robust Pareto Frontier (DRPO)')
    
    plt.title('DRPO: Feasible Objective Space and Robust Pareto Frontier', fontsize=14)
    plt.xlabel('Portfolio Risk ($\sigma$)', fontsize=12)
    plt.ylabel('Expected Annual Return ($E[R]$)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    print("\n--- Summary Table ---")
    print(pd.DataFrame({'Risk': frontier_risk, 'Return': frontier_ret}).tail(5))
    
    plt.show()